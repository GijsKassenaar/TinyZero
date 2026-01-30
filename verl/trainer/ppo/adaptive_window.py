#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import ast

import numpy as np
import torch

from verl import DataProto


@dataclass
class AdaptiveSuccessWindowConfig:
    """Configuration for adaptive max_tokens window based on successful trajectories."""

    enable: bool = False

    # Window bounds and initialization
    initial_window: int = 1024
    min_window: int = 20
    max_window: int = 12288

    # Warmup: number of PPO steps to use only for measuring typical success length.
    # During warmup, we do not change the window; after warmup, the initial
    # window is reset to the mean success length observed so far (clipped to
    # [min_window, max_window]) and normal adaptation begins from there.
    warmup_steps: int = 0

    # Statistics-based proposal
    std_multiplier: float = 2.0
    min_samples: int = 10
    update_frequency: int = 5  # in PPO steps
    min_change: int = 64

    # Success-rate driven curriculum
    target_success_rate: float = 0.7
    success_rate_margin: float = 0.05
    success_rate_ema_beta: float = 0.9
    growth_delta: int = 64
    shrink_delta: int = 64
    length_tolerance: int = 32
    hold_steps_before_growth: int = 20

    # Success definition
    success_threshold: float = 1.0

    # Mode: "basic" (all samples), "ema", "rolling", "phased"
    # - "basic": adaptive with all success samples
    # - "ema": adaptive with exponential moving average
    # - "rolling": adaptive with rolling window of samples
    # - "phased": simple step-based schedule (uses phased_schedule config)
    mode: str = "basic"
    rolling_window: int = 100

    # Exploration (epsilon-greedy)
    epsilon: float = 0.1

    # EMA parameters (only used when mode == "ema")
    ema_beta: float = 0.98
    ema_bias_correction: bool = True

    # Phased schedule (only used when mode == "phased")
    # Can be either:
    #   - List of [step_threshold, window_size] pairs: [[10, 512], [20, 1024]]
    #   - String representation: "[[10,512],[20,1024]]"
    # Example: [[10, 512], [20, 1024], [30, 2048]] means:
    #   - steps 0-9: 512 tokens
    #   - steps 10-19: 1024 tokens
    #   - steps 20-29: 2048 tokens
    #   - steps 30+: 2048 tokens (last window size is maintained)
    phased_schedule: Optional[Union[List[List[int]], str]] = None


def _parse_phased_schedule(schedule_input: Optional[Union[List[List[int]], str]]) -> Optional[List[List[int]]]:
    """Parse phased schedule from string or list format.
    
    Args:
        schedule_input: Either a list [[step, window], ...] or string "[[step,window],...]"
    
    Returns:
        Parsed list of [step, window] pairs, or None if input is None
    """
    if schedule_input is None:
        return None
    
    # If already a list, return as-is
    if isinstance(schedule_input, list):
        return schedule_input
    
    # If string, parse it
    if isinstance(schedule_input, str):
        try:
            # Use ast.literal_eval for safe parsing
            parsed = ast.literal_eval(schedule_input)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed)}")
            return parsed
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse phased_schedule string '{schedule_input}': {e}")
    
    raise TypeError(f"phased_schedule must be list or string, got {type(schedule_input)}")


class AdaptiveSuccessWindowController:
    """Adaptive controller for the generation window (max_tokens).

    Supports multiple modes:
    
    - Adaptive modes ("basic", "ema", "rolling"): 
      Tracks lengths of successful completions and adjusts window based on
      success rate and curriculum learning. Uses epsilon-greedy exploration.
    
    - Phased mode ("phased"):
      Simple step-based schedule where you specify exactly which window size
      to use at which training step. No adaptation, just follows the schedule.
    """

    def __init__(self, config: AdaptiveSuccessWindowConfig):
        self.config = config
        self.current_window: int = int(config.initial_window)

        # Parse and validate phased schedule configuration
        if config.mode == "phased":
            # Parse the schedule (handles both string and list formats)
            parsed_schedule = _parse_phased_schedule(config.phased_schedule)
            
            if parsed_schedule is None or len(parsed_schedule) == 0:
                print(f"Warning: phased mode enabled but phased_schedule is empty. Using initial_window={config.initial_window}")
                self._phased_schedule = []
            else:
                # Validate format of each phase
                for i, phase in enumerate(parsed_schedule):
                    if not isinstance(phase, (list, tuple)) or len(phase) != 2:
                        raise ValueError(f"phased_schedule[{i}] = {phase} is invalid. Expected [step, window] format.")
                    if not isinstance(phase[0], int) or not isinstance(phase[1], int):
                        raise ValueError(f"phased_schedule[{i}] = {phase} must contain integers [step, window].")
                
                # Store parsed schedule
                self._phased_schedule = parsed_schedule
                print(f"✓ Phased schedule initialized: {self._phased_schedule}")
        else:
            self._phased_schedule = []

        # Book-keeping
        self.step_count: int = 0
        self.total_samples: int = 0
        self.total_successes: int = 0
        self.total_failures: int = 0
        self.exploration_steps: int = 0
        self.cumulative_reward: float = 0.0

        # Success length buffer
        self._success_lengths: List[float] = []

        # EMA state (used when mode == "ema")
        self._ema_mean: Optional[float] = None
        self._ema_var: Optional[float] = None
        self._ema_count: int = 0

        # Success-rate tracking (for curriculum behaviour)
        self.success_rate_ema: Optional[float] = None
        self.last_change_step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_window_size(self) -> int:
        """Return the current max_tokens window."""
        return int(self.current_window)

    def update_from_batch(self, batch: DataProto, reward_tensor: torch.Tensor) -> Dict[str, float]:
        """Update controller statistics from a PPO batch and maybe adjust window.

        Args:
            batch: DataProto after rollout + reward computation.
            reward_tensor: token-level rewards, shape (batch, seq_len) or
                sequence-level rewards, shape (batch,).
        """
        if not self.config.enable:
            return {}

        self.step_count += 1

        # ------------------------------------------------------------------
        # Phased schedule mode: simple step-based window changes
        # ------------------------------------------------------------------
        if self.config.mode == "phased":
            if len(self._phased_schedule) > 0:
                # Find the appropriate window size for the current step
                # Schedule format: [[step, window], [step, window], ...]
                current_phase_window = self.config.initial_window
                for phase in self._phased_schedule:
                    step_threshold, window_size = phase[0], phase[1]
                    print(f"next: Step threshold: {step_threshold}, Window size: {window_size}")
                    if self.step_count >= step_threshold:
                        current_phase_window = window_size
                    else:
                        break  # Schedule should be sorted, so stop at first future threshold
                
                self.current_window = int(current_phase_window)
            
            # Return simple metrics for phased mode (no adaptive statistics needed)
            return {
                "adaptive_window/current_window": float(self.current_window),
                "adaptive_window/step_count": float(self.step_count),
                "adaptive_window/mode": 0.0,  # 0 = phased mode
            }

        # ------------------------------------------------------------------
        # Compute per-sequence rewards and response lengths
        # ------------------------------------------------------------------
        responses = batch.batch["responses"]
        response_length_tokens = responses.size(-1)

        attention_mask = batch.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length_tokens:]
        response_lengths = response_mask.sum(-1).float()  # (bs,)

        # Normalize reward_tensor to shape (bs,) if needed
        if reward_tensor.dim() == 2:
            seq_rewards = reward_tensor.sum(-1)
        else:
            seq_rewards = reward_tensor
        seq_rewards = seq_rewards.detach().float()

        batch_size = response_lengths.shape[0]
        self.total_samples += int(batch_size)

        # Success mask and lengths
        success_mask = seq_rewards >= self.config.success_threshold
        success_lengths = response_lengths[success_mask]
        num_success = int(success_lengths.numel())
        num_fail = int(batch_size - num_success)

        self.total_successes += num_success
        self.total_failures += num_fail
        self.cumulative_reward += float(seq_rewards.sum().item())

        if num_success > 0:
            # Update global success length buffer
            self._success_lengths.extend(success_lengths.cpu().tolist())

            # Update EMA statistics if enabled
            if self.config.mode == "ema":
                self._update_ema(success_lengths)

        # ------------------------------------------------------------------
        # Possibly update window (curriculum-style)
        # ------------------------------------------------------------------
        effective_lengths = self._get_effective_lengths()
        mean_len, median_len, std_len, p95_len, num_samples = self._compute_length_stats(effective_lengths)

        explored = False

        # Warmup phase: only collect statistics, do not adapt yet.
        if self.config.warmup_steps > 0 and self.step_count == self.config.warmup_steps:
            if num_samples > 0:
                warmup_window = float(np.clip(mean_len, self.config.min_window, self.config.max_window))
                self.current_window = int(round(warmup_window))
                self.last_change_step = self.step_count

        # Normal adaptive updates after warmup
        elif self.step_count > self.config.warmup_steps:
            if num_samples >= self.config.min_samples and (self.step_count % self.config.update_frequency == 0):
                new_window = self.current_window
                sr_ema = self.success_rate_ema if self.success_rate_ema is not None else batch_success_rate
                length_gap = self.current_window - mean_len

                high_sr = sr_ema >= (self.config.target_success_rate + self.config.success_rate_margin)
                low_sr = sr_ema <= (self.config.target_success_rate - self.config.success_rate_margin)
                near_len = abs(length_gap) <= self.config.length_tolerance

                # 1) Shrink when success rate is high and we're clearly over-provisioning tokens.
                if high_sr and length_gap > self.config.length_tolerance:
                    new_window = max(self.config.min_window, self.current_window - self.config.shrink_delta)

                # 2) Grow when success rate is low (we're struggling at this length).
                elif low_sr and self.current_window < self.config.max_window:
                    new_window = min(self.config.max_window, self.current_window + self.config.growth_delta)

                # 3) Curriculum growth: if stable and using most of the window, slowly expand.
                elif sr_ema >= self.config.target_success_rate and near_len and \
                        self.current_window < self.config.max_window and \
                        (self.step_count - self.last_change_step) >= self.config.hold_steps_before_growth:
                    new_window = min(self.config.max_window, self.current_window + self.config.growth_delta)

                # 4) Epsilon-greedy exploration toward max_window
                if np.random.rand() < self.config.epsilon and self.current_window < self.config.max_window:
                    new_window = int(self.config.max_window)
                    explored = True
                    self.exploration_steps += 1

                if abs(new_window - self.current_window) >= self.config.min_change:
                    self.current_window = int(new_window)
                    self.last_change_step = self.step_count

        # ------------------------------------------------------------------
        # Build metrics
        # ------------------------------------------------------------------
        batch_success_rate = float(num_success) / float(batch_size) if batch_size > 0 else 0.0

        # Update EMA success rate
        if self.success_rate_ema is None:
            self.success_rate_ema = batch_success_rate
        else:
            beta = self.config.success_rate_ema_beta
            self.success_rate_ema = beta * self.success_rate_ema + (1.0 - beta) * batch_success_rate

        cumulative_success_rate = (
            float(self.total_successes) / float(self.total_samples) if self.total_samples > 0 else 0.0
        )
        exploration_rate = (
            float(self.exploration_steps) / float(self.step_count) if self.step_count > 0 else 0.0
        )

        metrics: Dict[str, float] = {
            "adaptive_window/current_window": float(self.current_window),
            "adaptive_window/mean_success_length": float(mean_len),
            "adaptive_window/median_success_length": float(median_len),
            "adaptive_window/std_success_length": float(std_len),
            "adaptive_window/p95_success_length": float(p95_len),
            "adaptive_window/num_success_samples": float(num_samples),
            "adaptive_window/success_rate": batch_success_rate,
            "adaptive_window/success_rate_ema": float(self.success_rate_ema),
            "adaptive_window/exploration_rate": exploration_rate,
            "adaptive_window/epsilon": float(self.config.epsilon),
            "adaptive_window/cumulative_reward": float(self.cumulative_reward),
            "adaptive_window/cumulative_success_rate": float(cumulative_success_rate),
        }

        # It is sometimes useful for debugging to know if this step explored.
        metrics["adaptive_window/explored_step"] = 1.0 if explored else 0.0

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_effective_lengths(self) -> List[float]:
        """Return success lengths according to controller mode."""
        if len(self._success_lengths) == 0:
            return []

        if self.config.mode == "rolling":
            return self._success_lengths[-self.config.rolling_window :]

        # For "basic" and "ema" we use all collected lengths as the statistics base.
        return self._success_lengths

    def _update_ema(self, success_lengths: torch.Tensor) -> None:
        """Update EMA mean and variance from a tensor of success lengths."""
        if success_lengths.numel() == 0:
            return

        beta = self.config.ema_beta
        lengths = success_lengths.float()

        batch_mean = float(lengths.mean().item())
        batch_var = float(lengths.var(unbiased=False).item())

        if self._ema_mean is None:
            # Initialize EMA state
            self._ema_mean = batch_mean
            self._ema_var = batch_var
            self._ema_count = lengths.numel()
            return

        self._ema_mean = beta * self._ema_mean + (1.0 - beta) * batch_mean
        self._ema_var = beta * self._ema_var + (1.0 - beta) * batch_var
        self._ema_count += lengths.numel()

    def _compute_length_stats(self, lengths: List[float]):
        """Compute robust statistics on a list of lengths."""
        if not lengths:
            return 0.0, 0.0, 0.0, 0.0, 0

        arr = np.asarray(lengths, dtype=np.float32)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr))
        p95 = float(np.percentile(arr, 95))
        num = int(arr.size)

        # If EMA is enabled and has state, we prefer EMA stats for mean/std
        if self.config.mode == "ema" and self._ema_mean is not None and self._ema_var is not None:
            if self.config.ema_bias_correction:
                # Simple bias correction given number of updates
                correction = 1.0 - (self.config.ema_beta ** max(self._ema_count, 1))
                if correction > 0:
                    mean = float(self._ema_mean / correction)
                    std = float(np.sqrt(max(self._ema_var / correction, 0.0)))
            else:
                mean = float(self._ema_mean)
                std = float(np.sqrt(max(self._ema_var, 0.0)))

        return mean, median, std, p95, num


