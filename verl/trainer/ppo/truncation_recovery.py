"""
Truncation Recovery: Last-Chance Answer Inducer for Truncated Responses

When a response hits the generation budget (window size or max_response_length),
it's likely truncated mid-reasoning. This module gives truncated responses one
last chance to provide an answer by:

1. Detecting truncated samples (response_length >= generation_budget)
2. Pre-truncating them by N tokens before the limit
3. Appending an answer inducer prompt
4. Generating a short continuation to get the final answer
5. Replacing the truncated response with the recovered version
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List

import torch
from tensordict import TensorDict
from verl import DataProto


@dataclass
class TruncationRecoveryConfig:
    """Configuration for truncation recovery.
    
    recovery_window_tokens defines the window for recovery: we cut this many tokens
    from the end, add the inducer, and generate new tokens. The total new content
    (inducer + generated) will not exceed recovery_window_tokens, maintaining batch length.
    """
    enable: bool = False
    recovery_window_tokens: int = 50  # Tokens to cut and budget for recovery
    answer_inducer: str = "Time is limited, stop thinking and provide your final answer.\n</think>\n<answer>"


class TruncationRecoveryController:
    """Controls truncation recovery for responses that hit the generation budget."""
    
    def __init__(self, config: TruncationRecoveryConfig, tokenizer, max_response_length: int):
        self.config = config
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_token_id
        
        # Pre-tokenize answer inducer to calculate its length
        self.inducer_tokens = tokenizer.encode(config.answer_inducer, add_special_tokens=False)
        self.inducer_length = len(self.inducer_tokens)
        
        # Calculate max tokens we can generate while staying within recovery window
        # Total new content = inducer + generated must fit in recovery_window_tokens
        self.max_generation_tokens = max(1, config.recovery_window_tokens - self.inducer_length)
        
        print(f"[TruncationRecovery] Initialized with recovery_window={config.recovery_window_tokens}, "
              f"inducer_length={self.inducer_length}, max_gen_tokens={self.max_generation_tokens}")
        
        # Stats tracking
        self.total_truncated = 0
        self.total_recovered = 0
        self.total_samples = 0
    
    def get_truncated_mask(self, gen_batch_output: DataProto, generation_budget: int) -> torch.Tensor:
        """Identify which samples are truncated (hit the generation budget).
        
        Args:
            gen_batch_output: DataProto with generated responses
            generation_budget: The max tokens allowed during this generation
            
        Returns:
            Boolean tensor of shape (batch_size,) indicating truncated samples
        """
        responses = gen_batch_output.batch['responses']  # (batch_size, response_length)
        attention_mask = gen_batch_output.batch['attention_mask']  # (batch_size, total_length)
        
        batch_size = responses.shape[0]
        response_length = responses.shape[1]
        
        # Get the response portion of the attention mask
        response_mask = attention_mask[:, -response_length:]  # (batch_size, response_length)
        
        # Count valid tokens in response
        valid_response_lengths = response_mask.sum(dim=-1)  # (batch_size,)
        
        # A sample is truncated if it uses the full generation budget
        # (i.e., didn't naturally stop with EOS before the limit)
        truncated_mask = valid_response_lengths >= generation_budget
        
        return truncated_mask
    
    def recover_truncated_responses(
        self,
        gen_batch_output: DataProto,
        generation_budget: int,
        generate_fn: Callable,
    ) -> Tuple[DataProto, Dict[str, float]]:
        """Attempt to recover truncated responses with the answer inducer.
        
        Args:
            gen_batch_output: DataProto with generated responses
            generation_budget: The max tokens allowed during this generation
            generate_fn: Function to generate continuations (actor_rollout_wg.generate_sequences)
            
        Returns:
            Tuple of (modified DataProto with recovered responses, metrics dict)
        """
        truncated_mask = self.get_truncated_mask(gen_batch_output, generation_budget)
        num_truncated = truncated_mask.sum().item()
        batch_size = len(gen_batch_output.batch['responses'])
        
        self.total_samples += batch_size
        self.total_truncated += num_truncated
        
        metrics = {
            "truncation_recovery/truncated_count": num_truncated,
            "truncation_recovery/truncated_frac": num_truncated / batch_size if batch_size > 0 else 0.0,
            "truncation_recovery/inducer_length": self.inducer_length,
            "truncation_recovery/max_gen_tokens": self.max_generation_tokens,
        }
        
        if num_truncated == 0:
            # No truncated samples, return as-is
            metrics["truncation_recovery/recovered_count"] = 0
            metrics["truncation_recovery/recovered_frac"] = 0.0
            return gen_batch_output, metrics
        
        # Prepare prompts for truncated samples (with inducer appended)
        recovery_prompts, truncated_indices = self._prepare_recovery_prompts(
            gen_batch_output, truncated_mask, generation_budget
        )
        
        if recovery_prompts is None:
            metrics["truncation_recovery/recovered_count"] = 0
            metrics["truncation_recovery/recovered_frac"] = 0.0
            return gen_batch_output, metrics
        
        # Generate continuations for truncated samples
        # Use calculated budget that accounts for inducer tokens already present
        recovery_prompts.meta_info['max_tokens'] = self.max_generation_tokens
        recovery_outputs = generate_fn(recovery_prompts)
        
        # Merge recovered responses back into the original batch
        gen_batch_output = self._merge_recovered_responses(
            gen_batch_output, recovery_outputs, truncated_indices, generation_budget
        )
        
        self.total_recovered += len(truncated_indices)
        metrics["truncation_recovery/recovered_count"] = len(truncated_indices)
        metrics["truncation_recovery/recovered_frac"] = len(truncated_indices) / batch_size if batch_size > 0 else 0.0
        
        # Overall stats
        if self.total_truncated > 0:
            metrics["truncation_recovery/overall_recovery_rate"] = self.total_recovered / self.total_truncated
        
        print(f"[TruncationRecovery] Recovered {len(truncated_indices)}/{num_truncated} truncated samples "
              f"(recovery_window={self.config.recovery_window_tokens}, inducer_len={self.inducer_length}, max_gen={self.max_generation_tokens})")
        
        return gen_batch_output, metrics
    
    def _prepare_recovery_prompts(
        self,
        gen_batch_output: DataProto,
        truncated_mask: torch.Tensor,
        generation_budget: int
    ) -> Tuple[Optional[DataProto], List[int]]:
        """Create recovery prompts for truncated samples.
        
        For each truncated sample:
        1. Take original prompt + response[:budget - recovery_window_tokens]
        2. Append answer inducer tokens
        
        Returns:
            Tuple of (DataProto with recovery prompts, list of original batch indices)
        """
        prompts = gen_batch_output.batch['prompts']  # (batch_size, prompt_length)
        responses = gen_batch_output.batch['responses']  # (batch_size, response_length)
        device = prompts.device
        
        truncated_indices = truncated_mask.nonzero(as_tuple=True)[0].tolist()
        
        if not truncated_indices:
            return None, []
        
        # Calculate truncation point: cut recovery_window_tokens from the end
        truncate_at = generation_budget - self.config.recovery_window_tokens
        if truncate_at <= 0:
            print(f"[TruncationRecovery] Warning: recovery_window_tokens ({self.config.recovery_window_tokens}) "
                  f">= generation_budget ({generation_budget}). Skipping recovery.")
            return None, []
        
        all_input_ids = []
        all_attn_masks = []
        all_pos_ids = []
        
        inducer = torch.tensor(self.inducer_tokens, dtype=responses.dtype, device=device)
        
        for batch_idx in truncated_indices:
            prompt = prompts[batch_idx]
            response = responses[batch_idx]
            
            # Get the partial response (up to truncation point)
            # We need to handle padding - find the valid portion
            # For left-padded prompts, we need to find where the prompt actually starts
            prompt_valid = prompt[prompt != self.pad_token_id]
            
            # Take response up to truncate_at tokens (these are right-padded)
            partial_response = response[:truncate_at]
            
            # Concatenate: prompt + partial_response + inducer
            recovery_input = torch.cat([prompt_valid, partial_response, inducer], dim=0)
            length = len(recovery_input)
            
            all_input_ids.append(recovery_input)
            all_attn_masks.append(torch.ones(length, dtype=torch.long, device=device))
            all_pos_ids.append(torch.arange(length, dtype=torch.long, device=device))
        
        # Left-pad to same length (vLLM expects left-padded inputs)
        max_len = max(len(t) for t in all_input_ids)
        padded_ids, padded_masks, padded_pos = [], [], []
        
        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            if pad_len > 0:
                padded_ids.append(torch.cat([
                    torch.full((pad_len,), self.pad_token_id, dtype=all_input_ids[i].dtype, device=device),
                    all_input_ids[i]
                ]))
                padded_masks.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    all_attn_masks[i]
                ]))
                padded_pos.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    all_pos_ids[i]
                ]))
            else:
                padded_ids.append(all_input_ids[i])
                padded_masks.append(all_attn_masks[i])
                padded_pos.append(all_pos_ids[i])
        
        recovery_batch = TensorDict({
            'input_ids': torch.stack(padded_ids),
            'attention_mask': torch.stack(padded_masks),
            'position_ids': torch.stack(padded_pos),
        }, batch_size=len(truncated_indices))
        
        recovery_data = DataProto(
            batch=recovery_batch,
            non_tensor_batch={},
            meta_info=gen_batch_output.meta_info.copy() if hasattr(gen_batch_output, 'meta_info') else {}
        )
        
        return recovery_data, truncated_indices
    
    def _merge_recovered_responses(
        self,
        gen_batch_output: DataProto,
        recovery_outputs: DataProto,
        truncated_indices: List[int],
        generation_budget: int
    ) -> DataProto:
        """Merge recovered responses back into the original batch.
        
        The recovered response replaces the original truncated response by:
        1. Taking the partial reasoning (up to truncation point)
        2. Adding the inducer
        3. Adding the newly generated answer continuation
        4. Padding/truncating to match original response length
        """
        responses = gen_batch_output.batch['responses']  # (batch_size, response_length)
        input_ids = gen_batch_output.batch['input_ids']  # (batch_size, total_length)
        attention_mask = gen_batch_output.batch['attention_mask']  # (batch_size, total_length)
        prompts = gen_batch_output.batch['prompts']  # (batch_size, prompt_length)
        
        response_length = responses.shape[1]
        prompt_length = prompts.shape[1]
        device = responses.device
        
        truncate_at = generation_budget - self.config.recovery_window_tokens
        inducer = torch.tensor(self.inducer_tokens, dtype=responses.dtype, device=device)
        
        for i, batch_idx in enumerate(truncated_indices):
            # Get the newly generated continuation
            continuation = recovery_outputs.batch['responses'][i]
            
            # Build the new response: partial + inducer + continuation
            partial = responses[batch_idx, :truncate_at]
            new_response = torch.cat([partial, inducer, continuation], dim=0)
            
            # Pad or truncate to match original response length
            if len(new_response) < response_length:
                padding = torch.full(
                    (response_length - len(new_response),), 
                    self.pad_token_id, 
                    dtype=new_response.dtype, 
                    device=device
                )
                new_response = torch.cat([new_response, padding])
            else:
                new_response = new_response[:response_length]
            
            # Update the response in place
            responses[batch_idx] = new_response
            
            # Update input_ids (prompt + response)
            input_ids[batch_idx] = torch.cat([prompts[batch_idx], new_response])
            
            # Update attention mask for the response portion
            # Find EOS to mark end of valid tokens
            new_resp_mask = self._make_response_mask(new_response)
            attention_mask[batch_idx, prompt_length:] = new_resp_mask
        
        return gen_batch_output
    
    def _make_response_mask(self, response: torch.Tensor) -> torch.Tensor:
        """Create attention mask for a response, masking after EOS/PAD."""
        device = response.device
        mask = torch.ones(len(response), dtype=torch.long, device=device)
        
        # Find first EOS token
        eos_positions = (response == self.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            mask[first_eos + 1:] = 0
        
        # Also mask padding tokens (if pad != eos)
        if self.pad_token_id != self.eos_token_id:
            pad_positions = (response == self.pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                first_pad = pad_positions[0].item()
                mask[first_pad:] = 0
        
        return mask
    
    def get_stats(self) -> Dict[str, float]:
        """Get cumulative statistics."""
        return {
            "truncation_recovery/total_samples": self.total_samples,
            "truncation_recovery/total_truncated": self.total_truncated,
            "truncation_recovery/total_recovered": self.total_recovered,
            "truncation_recovery/overall_truncation_rate": (
                self.total_truncated / self.total_samples if self.total_samples > 0 else 0.0
            ),
            "truncation_recovery/overall_recovery_rate": (
                self.total_recovered / self.total_truncated if self.total_truncated > 0 else 0.0
            ),
        }
