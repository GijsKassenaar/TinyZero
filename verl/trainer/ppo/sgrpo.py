"""
S-GRPO: Serial-Group Decaying-Reward Policy Optimization

Two-phase generation:
1. Generate ONE complete reasoning path per prompt
2. Create K-1 truncated versions + generate continuations
3. Rewards decay exponentially: exit1=1.0, exit2=0.5, exit3=0.25, exit4=0.125
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict

import torch
from tensordict import TensorDict
from verl import DataProto


@dataclass
class SGRPOConfig:
    """S-GRPO configuration."""
    enable: bool = False
    num_exits: int = 4
    decay_factor: float = 2.0  # Reward divided by this for each later exit
    exit_method: str = "uniform"  # Only uniform supported for now
    answer_inducer: str = "Time is limited, stop thinking and start answering.\n</think>\n<answer>"


def get_uniform_exit_positions(response: torch.Tensor, num_exits: int, eos_token_id: int) -> List[int]:
    """Get uniformly spaced exit positions in a response."""
    # Find actual sequence length (up to EOS)
    eos_positions = (response == eos_token_id).nonzero(as_tuple=True)[0]
    length = eos_positions[0].item() + 1 if len(eos_positions) > 0 else len(response)
    
    if length < num_exits:
        return list(range(1, length + 1))
    
    return [max(1, min(int(length * (i + 1) / num_exits), length)) for i in range(num_exits)]


class SGRPOController:
    """Controls S-GRPO two-phase generation."""
    
    def __init__(self, config: SGRPOConfig, tokenizer, max_response_length: int):
        self.config = config
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_token_id
        
        # Pre-tokenize answer inducer
        self.inducer_tokens = tokenizer.encode(config.answer_inducer, add_special_tokens=False)
        
        # Stats tracking
        self.total_samples = 0
        self.correct_by_exit = {i: 0 for i in range(1, config.num_exits + 1)}
        self.total_by_exit = {i: 0 for i in range(1, config.num_exits + 1)}
    
    def create_serial_group_two_phase(
        self,
        full_responses: DataProto,
        generate_fn: Callable,
        max_new_tokens: int = 256
    ) -> Tuple[DataProto, torch.Tensor]:
        """Create serial group with two-phase generation.
        
        Phase 1 (done): full_responses has complete CoT
        Phase 2 (here): Create truncated prompts, generate continuations, combine
        """
        # Prepare truncated prompts
        truncated_prompts, exit_positions, orig_indices = self._prepare_truncated_prompts(full_responses)
        
        # Generate continuations
        truncated_continuations = None
        if truncated_prompts is not None:
            truncated_prompts.meta_info['max_tokens'] = max_new_tokens
            truncated_continuations = generate_fn(truncated_prompts)
        
        # Combine into serial group
        return self._combine_serial_group(full_responses, truncated_continuations, exit_positions)
    
    def _prepare_truncated_prompts(self, full_responses: DataProto) -> Tuple[Optional[DataProto], List[List[int]], List[int]]:
        """Create truncated prompts for Phase 2 generation."""
        batch_size = len(full_responses.batch['responses'])
        num_exits = self.config.num_exits
        
        prompts = full_responses.batch['prompts']
        responses = full_responses.batch['responses']
        device = prompts.device
        
        all_input_ids = []
        all_attn_masks = []
        all_pos_ids = []
        all_exit_positions = []
        all_orig_indices = []
        
        for batch_idx in range(batch_size):
            response = responses[batch_idx]
            prompt = prompts[batch_idx]
            
            exit_positions = get_uniform_exit_positions(response, num_exits, self.eos_token_id)
            all_exit_positions.append(exit_positions)
            
            # Create truncated prompts for exits 1 to K-1
            for exit_idx in range(num_exits - 1):
                exit_pos = exit_positions[exit_idx]
                
                partial = response[:exit_pos]
                inducer = torch.tensor(self.inducer_tokens, dtype=response.dtype, device=device)
                truncated = torch.cat([prompt, partial, inducer], dim=0)
                
                length = len(truncated)
                all_input_ids.append(truncated)
                all_attn_masks.append(torch.ones(length, dtype=torch.long, device=device))
                all_pos_ids.append(torch.arange(length, dtype=torch.long, device=device))
                all_orig_indices.append(batch_idx)
        
        if not all_input_ids:
            return None, all_exit_positions, []
        
        # Pad to same length (left-pad)
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
        
        truncated_batch = TensorDict({
            'input_ids': torch.stack(padded_ids),
            'attention_mask': torch.stack(padded_masks),
            'position_ids': torch.stack(padded_pos),
        }, batch_size=len(all_input_ids))
        
        truncated_data = DataProto(
            batch=truncated_batch,
            non_tensor_batch={},
            meta_info=full_responses.meta_info.copy() if hasattr(full_responses, 'meta_info') else {}
        )
        
        return truncated_data, all_exit_positions, all_orig_indices
    
    def _combine_serial_group(
        self,
        full_responses: DataProto,
        truncated_continuations: Optional[DataProto],
        exit_positions: List[List[int]]
    ) -> Tuple[DataProto, torch.Tensor]:
        """Combine full responses with truncated continuations."""
        batch_size = len(full_responses.batch['responses'])
        num_exits = self.config.num_exits
        device = full_responses.batch['responses'].device
        
        prompts = full_responses.batch['prompts']
        full_resp = full_responses.batch['responses']
        full_input_ids = full_responses.batch['input_ids']
        full_attn_mask = full_responses.batch['attention_mask']
        
        prompt_len = prompts.shape[1]
        resp_len = full_resp.shape[1]
        
        all_input_ids = []
        all_responses = []
        all_attn_masks = []
        all_prompts = []
        all_exit_orders = []
        
        truncated_idx = 0
        
        for batch_idx in range(batch_size):
            prompt = prompts[batch_idx]
            orig_resp = full_resp[batch_idx]
            orig_ids = full_input_ids[batch_idx]
            orig_mask = full_attn_mask[batch_idx]
            sample_exits = exit_positions[batch_idx]
            
            for exit_idx in range(num_exits):
                exit_order = exit_idx + 1
                
                if exit_idx == num_exits - 1:
                    # Full response
                    all_input_ids.append(orig_ids)
                    all_responses.append(orig_resp)
                    all_attn_masks.append(orig_mask)
                else:
                    # Truncated + continuation
                    exit_pos = sample_exits[exit_idx]
                    
                    if truncated_continuations is not None:
                        cont = truncated_continuations.batch['responses'][truncated_idx]
                        partial = orig_resp[:exit_pos]
                        inducer = torch.tensor(self.inducer_tokens, dtype=orig_resp.dtype, device=device)
                        
                        combined = torch.cat([partial, inducer, cont], dim=0)
                        
                        # Pad/truncate to resp_len
                        if len(combined) < resp_len:
                            combined = torch.cat([combined, torch.full(
                                (resp_len - len(combined),), self.pad_token_id, 
                                dtype=combined.dtype, device=device
                            )])
                        else:
                            combined = combined[:resp_len]
                        
                        combined_ids = torch.cat([prompt, combined])
                        combined_mask = self._make_attn_mask(combined, orig_mask[:prompt_len])
                        
                        all_input_ids.append(combined_ids)
                        all_responses.append(combined)
                        all_attn_masks.append(combined_mask)
                        truncated_idx += 1
                    else:
                        all_input_ids.append(orig_ids)
                        all_responses.append(orig_resp)
                        all_attn_masks.append(orig_mask)
                
                all_prompts.append(prompt)
                all_exit_orders.append(exit_order)
        
        serial_batch = TensorDict({
            'input_ids': torch.stack(all_input_ids),
            'responses': torch.stack(all_responses),
            'attention_mask': torch.stack(all_attn_masks),
            'prompts': torch.stack(all_prompts),
        }, batch_size=batch_size * num_exits)
        
        if 'position_ids' in full_responses.batch.keys():
            serial_batch['position_ids'] = self._make_position_ids(serial_batch['attention_mask'])
        
        serial_data = DataProto(
            batch=serial_batch,
            non_tensor_batch={},
            meta_info=full_responses.meta_info.copy() if hasattr(full_responses, 'meta_info') else {}
        )
        
        return serial_data, torch.tensor(all_exit_orders, dtype=torch.long, device=device)
    
    def _make_attn_mask(self, response: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        """Create attention mask for combined response."""
        device = response.device
        resp_mask = torch.ones(len(response), dtype=torch.long, device=device)
        
        # Mask after EOS
        eos_pos = (response == self.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            resp_mask[eos_pos[0].item() + 1:] = 0
        
        # Mask padding
        if self.pad_token_id != self.eos_token_id:
            pad_pos = (response == self.pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_pos) > 0:
                resp_mask[pad_pos[0].item():] = 0
        
        return torch.cat([prompt_mask, resp_mask])
    
    def _make_position_ids(self, attn_masks: torch.Tensor) -> torch.Tensor:
        """Create position IDs from attention masks."""
        batch_size, seq_len = attn_masks.shape
        device = attn_masks.device
        pos_ids = torch.zeros_like(attn_masks, dtype=torch.long)
        
        for i in range(batch_size):
            start = (attn_masks[i] == 1).nonzero(as_tuple=True)[0]
            if len(start) > 0:
                s = start[0].item()
                length = (attn_masks[i] == 1).sum().item()
                pos_ids[i, s:s+length] = torch.arange(length, device=device)
        
        return pos_ids
    
    def update_statistics(self, rewards: torch.Tensor, exit_orders: torch.Tensor) -> Dict[str, float]:
        """Update stats and return metrics for logging."""
        if rewards.dim() == 2:
            rewards = rewards.sum(dim=-1)
        
        is_correct = (rewards >= 0.5)
        
        for i in range(len(exit_orders)):
            idx = exit_orders[i].item()
            self.total_by_exit[idx] += 1
            if is_correct[i]:
                self.correct_by_exit[idx] += 1
        
        self.total_samples += len(exit_orders)
        
        metrics = {}
        for idx in range(1, self.config.num_exits + 1):
            if self.total_by_exit[idx] > 0:
                metrics[f"sgrpo/exit_{idx}_accuracy"] = self.correct_by_exit[idx] / self.total_by_exit[idx]
        
        correct_exits = exit_orders[is_correct].float()
        if len(correct_exits) > 0:
            metrics["sgrpo/avg_correct_exit_position"] = correct_exits.mean().item()
        
        if self.total_samples > 0:
            metrics["sgrpo/overall_accuracy"] = sum(self.correct_by_exit.values()) / self.total_samples
        
        return metrics
