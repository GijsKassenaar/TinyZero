# WandB Metrics Reference for TinyZero

This document describes all metrics logged to WandB during TinyZero training.

## Actor Metrics (`actor/*`)

| Metric | Description |
|--------|-------------|
| `actor/entropy_loss` | Average policy entropy; higher = more randomness in token selection |
| `actor/pg_loss` | PPO policy gradient loss; main objective being minimized |
| `actor/pg_clipfrac` | Fraction of tokens where PPO clipping was active; high = strong regularization |
| `actor/ppo_kl` | Approximate KL divergence π_new vs π_old; measures policy update magnitude |
| `actor/kl_loss` | Extra KL penalty term (when `use_kl_loss=True`); constrains deviation from reference |
| `actor/kl_coef` | Coefficient multiplying the KL loss term |
| `actor/is_ratio/mean` | Mean importance-sampling ratio π_new/π_old; should be near 1.0 |
| `actor/is_ratio/std` | Std dev of IS ratio; large values indicate unstable updates |
| `actor/is_ratio/p95` | 95th percentile of IS ratio; measures tail heaviness |
| `actor/is_log_ratio/mean` | Mean log importance-sampling ratio |
| `actor/is_log_ratio/std` | Std dev of log-IS ratio; dispersion of update sizes |
| `actor/is_clip/outside_frac` | Fraction of tokens where IS ratio is outside PPO clip range |
| `actor/grad_norm` | Global gradient norm; monitors gradient scale for stability |
| `actor/lr` | Current learning rate for the actor optimizer |

## Critic & Reward Metrics (`critic/*`)

| Metric | Description |
|--------|-------------|
| `critic/kl` | Current KL divergence between policy and reference (when using KL penalty) |
| `critic/kl_coeff` | Current KL penalty coefficient in reward shaping |
| `critic/score/mean` | Mean sequence-level raw task score (before KL shaping) |
| `critic/score/max` | Maximum raw task score in batch |
| `critic/score/min` | Minimum raw task score in batch |
| `critic/rewards/mean` | Mean total reward per sequence (after KL shaping if used) |
| `critic/rewards/max` | Maximum total reward in batch |
| `critic/rewards/min` | Minimum total reward in batch |
| `critic/advantages/mean` | Mean token-level advantage (drives actor updates) |
| `critic/advantages/max` | Maximum advantage value |
| `critic/advantages/min` | Minimum advantage value |
| `critic/returns/mean` | Mean token-level return (discounted cumulative reward) |
| `critic/returns/max` | Maximum return value |
| `critic/returns/min` | Minimum return value |
| `critic/values/mean` | Mean critic value prediction |
| `critic/values/max` | Maximum critic value |
| `critic/values/min` | Minimum critic value |
| `critic/vf_explained_var` | Value function explained variance; 1.0 = perfect predictions, 0.0 = no predictive power |

## Length & Token Statistics

| Metric | Description |
|--------|-------------|
| `response_length/mean` | Average number of response tokens per sample |
| `response_length/max` | Maximum response length in batch |
| `response_length/min` | Minimum response length in batch |
| `response_length/clip_ratio` | Fraction of responses hitting max_response_length (truncated) |
| `prompt_length/mean` | Average prompt length in tokens |
| `prompt_length/max` | Maximum prompt length |
| `prompt_length/min` | Minimum prompt length |
| `prompt_length/clip_ratio` | Fraction of prompts hitting max_prompt_length |
| `tokens/prompt_total` | Total prompt tokens in batch (summed) |
| `tokens/response_total` | Total response tokens in batch (summed) |
| `tokens/overall_total` | Total tokens processed (prompt + response) |

## Completion Metrics (`completion/*`)

These track whether responses finish naturally or hit the token budget.

| Metric | Description |
|--------|-------------|
| `completion/truncated_frac` | Fraction of samples that hit/exceeded the generation budget |
| `completion/finished_frac` | Fraction that finished before reaching the budget (with EOS) |
| `completion/truncated_correct_frac` | Among truncated samples, fraction with correct answers |
| `completion/finished_correct_frac` | Among finished samples, fraction with correct answers |
| `completion/correct_mean_length` | Mean response length of correct samples only |

## Difficulty Metrics (`difficulty/*`)

For datasets with difficulty labels (e.g., GSM8K levels 3 and 4).

| Metric | Description |
|--------|-------------|
| `difficulty/3_acc` | Accuracy on difficulty level 3 samples |
| `difficulty/3_count` | Number of level 3 samples in batch |
| `difficulty/4_acc` | Accuracy on difficulty level 4 samples |
| `difficulty/4_count` | Number of level 4 samples in batch |

## Adaptive Window Metrics (`adaptive_window/*`)

When adaptive window scheduling is enabled (`agent.adaptive_window.enable=True`).

| Metric | Description |
|--------|-------------|
| `adaptive_window/current_window` | Current generation budget (max response tokens) |
| `adaptive_window/mean_success_length` | Mean length of successful responses in batch |
| `adaptive_window/median_success_length` | Median length of successful responses |
| `adaptive_window/std_success_length` | Std dev of successful response lengths |
| `adaptive_window/p95_success_length` | 95th percentile length among successful responses |
| `adaptive_window/num_success_samples` | Number of successful samples used for window updates |
| `adaptive_window/success_rate` | Fraction of samples in this batch considered successful |
| `adaptive_window/success_rate_ema` | Exponential moving average of success rate |
| `adaptive_window/exploration_rate` | Fraction of steps where controller explored (jumped to max) |
| `adaptive_window/epsilon` | Current exploration probability parameter |
| `adaptive_window/cumulative_reward` | Cumulative sum of rewards over all steps |
| `adaptive_window/cumulative_success_rate` | Cumulative average success rate |
| `adaptive_window/explored_step` | 1.0 if this step explored, 0.0 otherwise |

## S-GRPO Metrics (`sgrpo/*`)

When S-GRPO (Serial-Group Decaying-Reward Policy Optimization) is enabled.

| Metric | Description |
|--------|-------------|
| `sgrpo/exit_1_accuracy` | Accuracy when exiting at position 1 (earliest) |
| `sgrpo/exit_2_accuracy` | Accuracy when exiting at position 2 |
| `sgrpo/exit_k_accuracy` | Accuracy for each exit position k (1 to num_exits) |
| `sgrpo/avg_correct_exit_position` | Average exit index among correct answers (lower = earlier answers) |
| `sgrpo/overall_accuracy` | Overall accuracy aggregated across all exits |

## Timing Metrics (`timing_s/*`, `timing_per_token_ms/*`)

Wall-clock timing for different phases of each training step.

| Metric | Description |
|--------|-------------|
| `timing_s/step` | Total time for one full PPO step (seconds) |
| `timing_s/gen` | Time spent in rollout generation |
| `timing_s/ref` | Time computing reference policy log-probs |
| `timing_s/values` | Time computing critic values |
| `timing_s/adv` | Time computing rewards and advantages |
| `timing_s/update_critic` | Time updating the critic network |
| `timing_s/update_actor` | Time updating the actor network |
| `timing_s/save_checkpoint` | Time saving model checkpoints |
| `timing_s/testing` | Time running validation |
| `timing_per_token_ms/gen` | Generation time per response token (milliseconds) |
| `timing_per_token_ms/ref` | Reference log-prob time per token |
| `timing_per_token_ms/values` | Value computation time per token |
| `timing_per_token_ms/adv` | Advantage computation time per token |
| `timing_per_token_ms/update_critic` | Critic update time per token |
| `timing_per_token_ms/update_actor` | Actor update time per token |
| `time/elapsed_s` | Cumulative wall-clock time since training start |

## Training Progress

| Metric | Description |
|--------|-------------|
| `train/cumulative_reward` | Cumulative sum of token-level rewards over all training steps |

---

## Entropy Logging (Local Disk Only)

**Per-token entropy and varentropy are NOT logged to WandB.** When `agent.entropy_logging.enable=True`, raw entropy data is saved locally to:

```
{default_local_dir}/entropy_data/entropy_step_{step:06d}.pt
```

Each `.pt` file contains:
- `old_entropy`: (batch_size, response_len) - Shannon entropy per token
- `old_varentropy`: (batch_size, response_len) - Variance of log probabilities per token
- `attention_mask`: (batch_size, seq_len) - Mask identifying valid tokens
- `responses`: (batch_size, response_len) - Token IDs

This data must be analyzed separately using custom scripts.

---

## Notes

- All metrics are computed per training step and logged at the end of each step
- Metrics with `/mean`, `/max`, `/min` are statistics over the batch
- Token-level metrics (advantages, returns, etc.) are masked to only count valid (non-padding) response tokens
- Timing metrics help identify bottlenecks in the training pipeline
- For GRPO runs (`algorithm.adv_estimator=grpo`), there are `n` rollouts per prompt, where `n` is set by `actor_rollout_ref.rollout.n`
