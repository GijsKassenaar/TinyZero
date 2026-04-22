from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AdaptiveGroupBudgetConfig:
    """Runtime configuration for adaptive two-stage GRPO group budgeting."""

    enable: bool = False
    initial_group_size: int = 2
    budget_group_size: Optional[int] = None
    correct_threshold: float = 0.5
    remainder_policy: str = "round_robin"
    max_group_size: Optional[int] = None


class AdaptiveGroupBudgetController:
    """Allocates stage-2 rollout budget over hard prompts deterministically."""

    def __init__(self, config: AdaptiveGroupBudgetConfig):
        self.config = config

        if self.config.initial_group_size <= 0:
            raise ValueError("adaptive_group_budget.initial_group_size must be positive")

        if self.config.remainder_policy != "round_robin":
            raise ValueError(
                "adaptive_group_budget.remainder_policy only supports 'round_robin' in v1"
            )

        if self.config.max_group_size is not None and self.config.max_group_size < self.config.initial_group_size:
            raise ValueError(
                "adaptive_group_budget.max_group_size must be >= adaptive_group_budget.initial_group_size"
            )

    def get_initial_group_size(self) -> int:
        return int(self.config.initial_group_size)

    def get_budget_group_size(self, fallback_budget_group_size: int) -> int:
        budget_group_size = (
            int(self.config.budget_group_size)
            if self.config.budget_group_size is not None
            else int(fallback_budget_group_size)
        )
        if budget_group_size < self.config.initial_group_size:
            raise ValueError(
                "adaptive_group_budget budget group size must be >= initial_group_size, "
                f"got {budget_group_size} and {self.config.initial_group_size}"
            )
        return budget_group_size

    def compute_stage_two_allocation(
        self,
        num_prompts: int,
        hard_prompt_mask: np.ndarray,
        fallback_budget_group_size: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Compute per-prompt stage-2 extra rollout counts.

        Returns:
            extras_per_prompt: shape (num_prompts,), integer extra rollout count per prompt.
            metrics: allocation summary metrics.
        """
        if hard_prompt_mask.dtype != bool:
            raise ValueError(f"hard_prompt_mask must be boolean, got dtype={hard_prompt_mask.dtype}")
        if hard_prompt_mask.shape[0] != num_prompts:
            raise ValueError(
                f"hard_prompt_mask length {hard_prompt_mask.shape[0]} does not match num_prompts {num_prompts}"
            )

        initial_group_size = int(self.config.initial_group_size)
        budget_group_size = self.get_budget_group_size(fallback_budget_group_size)

        total_budget_rollouts = int(num_prompts * budget_group_size)
        stage1_rollouts = int(num_prompts * initial_group_size)
        remaining_rollouts = int(max(total_budget_rollouts - stage1_rollouts, 0))

        hard_indices = np.nonzero(hard_prompt_mask)[0]
        hard_prompt_count = int(hard_indices.shape[0])

        extras_per_prompt = np.zeros(num_prompts, dtype=np.int64)

        if remaining_rollouts > 0 and hard_prompt_count > 0:
            base_extra = int(remaining_rollouts // hard_prompt_count)
            remainder = int(remaining_rollouts % hard_prompt_count)

            hard_extras = np.full(hard_prompt_count, base_extra, dtype=np.int64)
            if remainder > 0:
                # Deterministic round-robin remainder assignment in stable hard prompt order.
                hard_extras[:remainder] += 1

            if self.config.max_group_size is not None:
                max_extra_per_prompt = max(int(self.config.max_group_size) - initial_group_size, 0)

                hard_extras = np.minimum(hard_extras, max_extra_per_prompt)

                residual = int(remaining_rollouts - hard_extras.sum())
                if residual > 0 and max_extra_per_prompt > 0:
                    expandable = hard_extras < max_extra_per_prompt
                    while residual > 0 and np.any(expandable):
                        expandable_indices = np.nonzero(expandable)[0]
                        for idx in expandable_indices:
                            if residual == 0:
                                break
                            hard_extras[idx] += 1
                            residual -= 1
                            if hard_extras[idx] >= max_extra_per_prompt:
                                expandable[idx] = False

            extras_per_prompt[hard_indices] = hard_extras

        final_group_sizes = np.full(num_prompts, initial_group_size, dtype=np.int64) + extras_per_prompt
        hard_prompt_frac = float(hard_prompt_count / max(num_prompts, 1))

        # Keep only compact, high-signal metrics that vary during training.
        metrics = {
            "adaptive_group_budget/hard_prompt_frac": hard_prompt_frac,
            "adaptive_group_budget/easy_prompt_frac": float(1.0 - hard_prompt_frac),
            "adaptive_group_budget/mean_group_size": float(final_group_sizes.mean()) if num_prompts > 0 else 0.0,
            "adaptive_group_budget/max_group_size": float(final_group_sizes.max()) if num_prompts > 0 else 0.0,
        }

        return extras_per_prompt, metrics

    @staticmethod
    def build_stage_two_prompt_indices(extras_per_prompt: np.ndarray) -> np.ndarray:
        """Expand prompt indices by per-prompt stage-2 extra counts."""
        if extras_per_prompt.ndim != 1:
            raise ValueError(f"extras_per_prompt must be 1D, got shape={extras_per_prompt.shape}")
        prompt_indices = np.arange(extras_per_prompt.shape[0], dtype=np.int64)
        return np.repeat(prompt_indices, extras_per_prompt.astype(np.int64))
