# Copyright 2026 TinyZero Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

import torch

from verl import DataProto


_THINKING_CLOSE_TAG_PATTERN = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)


def _tokenize_text_length(text: str, tokenizer: Any) -> int:
    """Return token length for text without adding special tokens."""
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return len(tokenized["input_ids"])


def _extract_reasoning_spans(response_text: str) -> tuple[list[str], bool]:
    """Extract reasoning text for prompt-prefilled <think> format.

    The opening think tag is expected in the prompt prefix, so response text starts
    inside think and reasoning ends at the first closing think tag.
    """
    close_match = _THINKING_CLOSE_TAG_PATTERN.search(response_text)
    if close_match is not None:
        prefix = response_text[: close_match.start()]
        if prefix.strip():
            return [prefix], True

        return [], True

    return [], False


def compute_reasoning_token_statistics(
    batch: DataProto,
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reasoning-token mask/statistics from prompt-prefilled think responses.

    Returns:
        reasoning_mask: Bool tensor of shape (bs, response_len_max) with reasoning tokens set True.
        reasoning_lengths: Float tensor of shape (bs,) containing reasoning token counts K.
        closed_think: Bool tensor of shape (bs,) indicating whether a closing think tag was found.
        valid_response_lengths: Long tensor of shape (bs,) with valid response lengths from attention mask.
    """
    responses = batch.batch["responses"]
    response_len_max = responses.size(1)
    response_mask = batch.batch["attention_mask"][:, -response_len_max:]
    valid_response_lengths = response_mask.sum(-1).to(dtype=torch.long)

    reasoning_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    reasoning_lengths = torch.zeros(responses.size(0), dtype=torch.float32, device=responses.device)
    closed_think = torch.zeros(responses.size(0), dtype=torch.bool, device=responses.device)

    for i in range(responses.size(0)):
        valid_len = int(valid_response_lengths[i].item())
        if valid_len <= 0:
            continue

        response_ids = responses[i, :valid_len].tolist()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        spans, has_close_tag = _extract_reasoning_spans(response_text)
        closed_think[i] = bool(has_close_tag)
        if not has_close_tag:
            continue

        reasoning_token_count = 0
        for span in spans:
            reasoning_token_count += _tokenize_text_length(span, tokenizer)

        reasoning_token_count = min(int(reasoning_token_count), valid_len)
        reasoning_lengths[i] = float(reasoning_token_count)
        if reasoning_token_count > 0:
            reasoning_mask[i, :reasoning_token_count] = True

    return reasoning_mask, reasoning_lengths, closed_think, valid_response_lengths


def apply_reasoning_reward_discount(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    tokenizer: Any,
    discount_cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply exponential discount gamma^K where K is reasoning tokens inside think tags."""
    gamma = float(discount_cfg.get("gamma", 1.0))
    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError(f"discounted_reasoning.gamma must satisfy 0 < gamma <= 1, got {gamma}")

    _, reasoning_lengths_tensor, closed_think_tensor, valid_response_lengths = compute_reasoning_token_statistics(
        batch=batch,
        tokenizer=tokenizer,
    )
    reasoning_lengths_tensor = reasoning_lengths_tensor.to(device=reward_tensor.device, dtype=torch.float32)
    closed_think_tensor = closed_think_tensor.to(device=reward_tensor.device, dtype=torch.bool)
    valid_response_lengths = valid_response_lengths.to(device=reward_tensor.device, dtype=torch.long)
    valid_reasoning_mask = (valid_response_lengths > 0) & closed_think_tensor
    discount_factors = torch.pow(
        torch.full_like(reasoning_lengths_tensor, gamma, dtype=torch.float32),
        reasoning_lengths_tensor,
    )

    discounted_reward_tensor = reward_tensor * discount_factors.unsqueeze(-1)

    if valid_reasoning_mask.any():
        mean_reasoning_tokens = float(reasoning_lengths_tensor[valid_reasoning_mask].mean().item())
        mean_discount_factor = float(discount_factors[valid_reasoning_mask].mean().item())
    else:
        mean_reasoning_tokens = 0.0
        mean_discount_factor = 0.0

    metrics = {
        "discounted_reasoning/gamma": gamma,
        "discounted_reasoning/mean_reasoning_tokens": mean_reasoning_tokens,
        "discounted_reasoning/mean_discount_factor": mean_discount_factor,
    }
    return discounted_reward_tensor, metrics
