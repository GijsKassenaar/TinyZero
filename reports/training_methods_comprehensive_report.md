# TinyZero Comprehensive Methods Report

Date: 2026-04-14

This report is a full method audit built from all training launchers under scripts and traced into implementation code.
It is intended as a working research notebook you can extend with outcomes later.

## Scope and Coverage

Total launchers reviewed: 26

Covered files:
- scripts/scripts2/train_tiny_zero_adaptive.job
- scripts/scripts2/train_tiny_zero_adaptive.sh
- scripts/scripts2/train_tiny_zero_cosine.sh
- scripts/scripts2/train_tiny_zero_full.job
- scripts/scripts2/train_tiny_zero_phased_schedule.sh
- scripts/scripts2/train_tiny_zero_phased.sh
- scripts/scripts2/train_tiny_zero.sh
- scripts/scripts2/train_tiny_zero_test.job
- scripts/scripts2/train_tiny_zero_vanilla.job
- scripts/scripts2/train_tiny_zero_vanilla.sh
- scripts/sgrpo/train_tiny_zero_hybrid_sgrpo.job
- scripts/sgrpo/train_tiny_zero_sgrpo.job
- scripts/sgrpo/train_tiny_zero_sgrpo_warmup.job
- scripts/train_dapo_test.job
- scripts/train_gamma_additive_tau_test.job
- scripts/train_gamma_test.job
- scripts/train_grpo_lambda_flat_incorrect_trace_test.job
- scripts/train_grpo_lambda_group_shortest_trace_test.job
- scripts/train_grpo_lambda_second_trace_test.job
- scripts/train_grpo_lambda_test.job
- scripts/train_grpo_lambda_token_normalization_additive_tau_test.job
- scripts/train_grpo_lambda_token_normalization_test.job
- scripts/train_lambda_gamma.job
- scripts/train_lead_grpo_lambda_test.job
- scripts/train_lead_grpo_test.job
- scripts/train_normalphased_test.job

Code paths traced:
- verl/trainer/config/algorithm.py
- verl/trainer/config/ppo_trainer.yaml
- verl/trainer/ppo/ray_trainer.py
- verl/trainer/ppo/core_algos.py
- verl/trainer/ppo/adaptive_window.py
- verl/trainer/ppo/sgrpo.py
- verl/trainer/ppo/discounted_reasoning.py
- verl/trainer/ppo/metric_utils.py
- verl/workers/utils/losses.py
- recipe/dapo/main_dapo.py
- recipe/dapo/dapo_ray_trainer.py
- verl/workers/reward_manager/dapo.py

## Method 1: Standard GRPO Phased Baseline

Scripts:
- scripts/train_normalphased_test.job

Script-level additions:
- algorithm.adv_estimator=grpo
- actor_rollout_ref.rollout.n=4
- agent.adaptive_window.enable=True with mode=phased and schedule [[50,2048]]
- trainer.resume_mode=resume_path and trainer.val_only=True

What code does:
- Group-relative outcome advantages are computed in compute_grpo_outcome_advantage in verl/trainer/ppo/core_algos.py.
- Grouping uses uid, then per-group centering and optional std normalization.
- Phased adaptive window is handled by AdaptiveSuccessWindowController in verl/trainer/ppo/adaptive_window.py, but this script is val_only so fit exits after validation.

Behavioral interpretation:
- This launcher behaves more like a resumed validation pass than active training.
- Good for quick checkpoint checks under the same config surface as phased GRPO.

Results (to fill later):
- Run IDs / checkpoint path:
- Validation accuracy:
- Completion metrics (truncated or finished):
- Notes:

## Method 2: Fixed-Length Baselines (Legacy scripts2 family)

Scripts:
- scripts/scripts2/train_tiny_zero.sh
- scripts/scripts2/train_tiny_zero_full.job
- scripts/scripts2/train_tiny_zero_vanilla.sh
- scripts/scripts2/train_tiny_zero_vanilla.job

Script-level additions:
- Full baseline: data.max_response_length=4096, no adaptive window settings.
- Vanilla baseline: data.max_response_length=1024 and agent.adaptive_window.enable=False.
- Wrapper job files only export env vars and delegate to shell launchers.

What code does:
- With no adaptive controller enabled, rollout budget is fixed by response length and rollout config.
- GRPO advantage path remains the default if adv_estimator is not changed.

Behavioral interpretation:
- Full baseline tests long-context fixed rollouts.
- Vanilla baseline removes adaptive curriculum and fixes shorter outputs.

Caveat:
- scripts/scripts2/*.job wrappers call scripts/train_tiny_zero*.sh paths, while current files live in scripts/scripts2/. These wrappers may need path correction before use.

Results (to fill later):
- Run IDs:
- Accuracy vs response length:
- Token cost vs accuracy:
- Notes:

## Method 3: Adaptive Window Controller (Dynamic max_tokens)

Scripts:
- scripts/scripts2/train_tiny_zero_adaptive.sh
- scripts/scripts2/train_tiny_zero_adaptive.job
- scripts/scripts2/train_tiny_zero_test.job
- scripts/train_normalphased_test.job
- scripts/train_gamma_test.job
- scripts/train_gamma_additive_tau_test.job
- scripts/train_grpo_lambda_flat_incorrect_trace_test.job
- scripts/train_grpo_lambda_group_shortest_trace_test.job
- scripts/train_grpo_lambda_second_trace_test.job
- scripts/train_grpo_lambda_token_normalization_test.job
- scripts/train_grpo_lambda_token_normalization_additive_tau_test.job
- scripts/train_lambda_gamma.job
- scripts/train_lead_grpo_test.job
- scripts/train_lead_grpo_lambda_test.job
- scripts/sgrpo/train_tiny_zero_sgrpo_warmup.job
- scripts/sgrpo/train_tiny_zero_hybrid_sgrpo.job

Script-level additions:
- agent.adaptive_window.enable=True
- Key knobs used: initial_window, min_window, max_window, mode, phased_schedule, success_threshold, epsilon

What code does:
- Controller is initialized in RayPPOTrainer.__init__ in verl/trainer/ppo/ray_trainer.py.
- Current max_tokens is injected before generation each step.
- update_from_batch in verl/trainer/ppo/adaptive_window.py updates window from success lengths and success rate.
- In phased mode it follows fixed thresholds, in adaptive modes it shrinks, grows, or explores based on success-rate logic.

Behavioral interpretation:
- Moves generation budget from static to feedback-driven.
- Enables curriculum pressure on reasoning length without changing the reward function itself.

Caveat:
- scripts/scripts2/train_tiny_zero_test.job has total_training_steps=6 but phased_schedule switch at step 50, so no actual phase switch occurs in that run.

Results (to fill later):
- Run IDs:
- Window trajectory over training:
- Success-rate trajectory:
- Accuracy and token efficiency:
- Notes:

## Method 4: Shell-Orchestrated Length Schedules (Not trainer-internal)

Scripts:
- scripts/scripts2/train_tiny_zero_phased.sh
- scripts/scripts2/train_tiny_zero_phased_schedule.sh
- scripts/scripts2/train_tiny_zero_cosine.sh

Script-level additions:
- These scripts loop over phases and relaunch main_ppo with different data.max_response_length values.
- Cosine script computes phase lengths via an inline Python cosine ramp.

What code does:
- No special algorithm module in trainer is activated.
- Scheduling is external orchestration by shell control flow.

Behavioral interpretation:
- Easy way to test curricula without touching trainer internals.
- But phases are separate launches unless explicit checkpoint chaining is added.

Results (to fill later):
- Run IDs per phase:
- Phase transition strategy used:
- Accuracy and cost per phase:
- Notes:

## Method 5: Discounted Reasoning Rewards (gamma over reasoning tokens)

Scripts:
- scripts/train_gamma_test.job
- scripts/train_gamma_additive_tau_test.job

Script-level additions:
- algorithm.discounted_reasoning.enable=True
- algorithm.discounted_reasoning.gamma=0.999

What code does:
- Reasoning token spans are inferred in verl/trainer/ppo/discounted_reasoning.py by detecting text before first closing think tag.
- In ray_trainer compute_advantage, reasoning lengths are computed when enabled.
- In core_algos compute_grpo_outcome_advantage and compute_grpo_lambda_advantages, correct-sequence rewards are multiplied by gamma^K.

Behavioral interpretation:
- Penalizes very long reasoning on successful samples while keeping binary correctness objective.
- Tightens reward scale for long successful traces and can alter group normalization dynamics.

Results (to fill later):
- Run IDs:
- Mean reasoning tokens:
- Mean discount factor:
- Accuracy vs baseline:
- Notes:

## Method 6: Plain GRPO Additive Normalization (std + tau)

Scripts:
- scripts/train_gamma_additive_tau_test.job

Script-level additions:
- algorithm.grpo_additive_normalization_enable=True
- algorithm.grpo_additive_normalization_tau=$TAU (array sweep)

What code does:
- In compute_grpo_outcome_advantage in core_algos, normalization denominator switches from std+epsilon to std+tau.

Behavioral interpretation:
- Reduces extreme scaling when per-group std is tiny.
- Intended to stabilize updates in low-variance reward regimes.

Results (to fill later):
- Tau values tested:
- Stability indicators (loss spikes, grad norms):
- Accuracy and token metrics:
- Notes:

## Method 7: Global Incorrect-Answer Penalty

Scripts:
- scripts/train_gamma_additive_tau_test.job
- scripts/train_lead_grpo_test.job
- scripts/train_lead_grpo_lambda_test.job

Script-level additions:
- algorithm.incorrect_answer_penalty.enable=True
- algorithm.incorrect_answer_penalty.penalty=-1.0

What code does:
- In GRPO and GRPO-lambda code paths, incorrect sequence rewards are replaced by fixed penalty before group normalization.
- Implemented in compute_grpo_outcome_advantage and compute_grpo_lambda_advantages in core_algos.

Behavioral interpretation:
- Makes incorrect samples consistently punitive instead of near-zero.
- Increases separation between correct and incorrect trajectories.

Results (to fill later):
- Run IDs:
- Incorrect fraction and calibration impact:
- Effect on stability and convergence speed:
- Notes:

## Method 8: LEAD Shaping for GRPO and GRPO-lambda

Scripts:
- scripts/train_lead_grpo_test.job
- scripts/train_lead_grpo_lambda_test.job

Script-level additions:
- algorithm.lead.enable=True
- algorithm.lead.alpha, tau, beta, epsilon
- Combined with incorrect_answer_penalty in both scripts

What code does:
- Implemented in compute_grpo_outcome_advantage in core_algos.
- Correct samples receive length z-score shaping via exp(-alpha * z).
- Group-level weight uses Gaussian around target accuracy tau with width beta.
- In lead+lambda launcher, this combines with grpo_lambda actor loss mode and lambda advantage pipeline.

Behavioral interpretation:
- Injects explicit difficulty-aware and length-aware shaping.
- Can favor shorter correct solutions when length-normalized z-score is high.

Caveat:
- scripts/train_lead_grpo_test.job uses trainer.resume_from_path=auto\ and trainer.total_training_steps=0, which may indicate an execution typo or no-op training.

Results (to fill later):
- Run IDs:
- Per-difficulty accuracy changes:
- Length distribution shift:
- Notes:

## Method 9: GRPO-lambda Baseline Stack

Scripts:
- scripts/train_grpo_lambda_test.job
- scripts/train_grpo_lambda_flat_incorrect_trace_test.job
- scripts/train_grpo_lambda_group_shortest_trace_test.job
- scripts/train_grpo_lambda_second_trace_test.job
- scripts/train_grpo_lambda_token_normalization_test.job
- scripts/train_grpo_lambda_token_normalization_additive_tau_test.job
- scripts/train_lambda_gamma.job
- scripts/train_lead_grpo_lambda_test.job

Script-level additions:
- actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda
- algorithm.adv_estimator=grpo_lambda
- critic.enable=False in GRPO-lambda test scripts

What code does:
- Policy loss mode is selected via get_policy_loss_fn in verl/workers/utils/losses.py.
- GRPO-lambda policy objective is registered as compute_policy_loss_grpo_lambda in core_algos.
- Advantages come from compute_grpo_lambda_outcome_advantage, which groups by uid and applies token-level trace logic.

Behavioral interpretation:
- Couples grouped outcome normalization with token-level eligibility trace credit assignment.
- Enables variant controls through algorithm.grpo_lambda_variant.*.

Results (to fill later):
- Run IDs:
- Baseline lambda metrics:
- Comparison anchor for all lambda variants:
- Notes:

## Method 10: GRPO-lambda Variant A - Flat Incorrect Trace

Scripts:
- scripts/train_grpo_lambda_flat_incorrect_trace_test.job

Script-level additions:
- algorithm.grpo_lambda_variant.enable=True
- algorithm.grpo_lambda_variant.flat_incorrect_trace=True

What code does:
- In compute_grpo_lambda_advantages, incorrect samples use flat trace instead of decayed positional trace.
- Correct samples keep normal decay trace.

Behavioral interpretation:
- Spreads corrective signal across incorrect trajectory tokens rather than emphasizing late positions.

Results (to fill later):
- Run IDs:
- Incorrect trajectory learning behavior:
- Accuracy and length impact:
- Notes:

## Method 11: GRPO-lambda Variant B - Sequence Gamma Discount + Reasoning-only Trace

Scripts:
- scripts/train_lambda_gamma.job

Script-level additions:
- algorithm.grpo_lambda_variant.sequence_gamma_discount_enable=True
- algorithm.grpo_lambda_variant.sequence_discount_gamma=0.99999999
- algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=True

What code does:
- In compute_grpo_lambda_advantages, sequence reward is discounted by gamma^K using trace token lengths.
- Trace weighting can be restricted to reasoning token mask only.
- Reasoning mask is generated from think-tag boundary logic in discounted_reasoning utilities.

Behavioral interpretation:
- Separates "how much" reward (sequence discount) from "where" credit goes (reasoning-only trace).

Results (to fill later):
- Run IDs:
- Reasoning-length sensitivity:
- Accuracy and token efficiency:
- Notes:

## Method 12: GRPO-lambda Variant C - Token Normalization (with post-EOS carry)

Scripts:
- scripts/train_grpo_lambda_token_normalization_test.job

Script-level additions:
- algorithm.grpo_lambda_variant.token_normalization_enable=True

What code does:
- In compute_grpo_lambda_advantages, outcome is traced first then normalized per timestep across rollouts in a group.
- Padded positions carry terminal signal naturally through trace exponents and masking scheme.

Behavioral interpretation:
- Can stabilize token-level credit by controlling per-position outliers.
- Changes normalization geometry from sequence-level to timestep-level.

Results (to fill later):
- Run IDs:
- Token-level variance diagnostics:
- Accuracy and stability:
- Notes:

## Method 13: GRPO-lambda Variant D - Token-Norm Additive Tau

Scripts:
- scripts/train_grpo_lambda_token_normalization_additive_tau_test.job

Script-level additions:
- token_normalization_enable=True
- additive_normalization_enable=True
- additive_normalization_tau=0.02

What code does:
- In token normalization branch, denominator is std+tau instead of std+epsilon.

Behavioral interpretation:
- Dampens very high normalized values when per-timestep std is near zero.

Results (to fill later):
- Run IDs:
- Stability and clip fraction trends:
- Accuracy and token metrics:
- Notes:

## Method 14: GRPO-lambda Variant E - Second Trace After Token Norm

Scripts:
- scripts/train_grpo_lambda_second_trace_test.job

Script-level additions:
- token_normalization_enable=True
- second_trace_after_token_norm_enable=True
- second_trace_alpha=10

What code does:
- After token normalization, a second positional trace multiplier decay_trace^alpha is applied.

Behavioral interpretation:
- Adds another degree of temporal shaping after normalization.
- High alpha can strongly reweight early/late tokens depending on trace profile.

Results (to fill later):
- Run IDs:
- Position-wise advantage distribution:
- Accuracy and length behavior:
- Notes:

## Method 15: GRPO-lambda Variant F - Group-Shortest Trace Base

Scripts:
- scripts/train_grpo_lambda_group_shortest_trace_test.job

Script-level additions:
- group_shortest_lambda_enable=True
- group_shortest_lambda_alpha=0.15

What code does:
- Shared decay base per group is calibrated from shortest valid response length and alpha.
- All samples in group share this base for trace geometry.

Behavioral interpretation:
- Reduces trace-shape mismatch between long and short group members.
- Explicitly targets length-robust credit assignment.

Results (to fill later):
- Run IDs:
- Length-bias diagnostics:
- Accuracy and variance metrics:
- Notes:

## Method 16: S-GRPO Core (Serial exits with decayed rewards)

Scripts:
- scripts/sgrpo/train_tiny_zero_sgrpo.job

Script-level additions:
- algorithm.adv_estimator=sgrpo
- algorithm.sgrpo.enable=True
- algorithm.sgrpo.num_exits=4
- algorithm.sgrpo.decay_factor=2.0
- algorithm.sgrpo.exit_method=uniform
- critic.enable=False

What code does:
- SGRPOController in verl/trainer/ppo/sgrpo.py creates truncated exits plus continuation generation.
- compute_sgrpo_outcome_advantage in core_algos applies reward decay by exit order for correct samples.
- During active S-GRPO phase, ray_trainer sets rollout_repeat_times to 1 and handles serial groups explicitly.

Behavioral interpretation:
- Encourages earlier correct exits and penalizes unnecessary long reasoning.

Results (to fill later):
- Run IDs:
- Exit-wise accuracy:
- Avg correct exit position:
- Notes:

## Method 17: S-GRPO Warmup + Phased Window

Scripts:
- scripts/sgrpo/train_tiny_zero_sgrpo_warmup.job

Script-level additions:
- algorithm.sgrpo.warmup_steps=50
- adaptive window phased from 1024 to 2048 at step 50

What code does:
- Before warmup threshold, ray_trainer maps sgrpo estimator to GRPO for advantage computation.
- After threshold, S-GRPO serial generation path activates.

Behavioral interpretation:
- Uses GRPO-style stabilization before turning on serial-exit pressure.

Results (to fill later):
- Run IDs:
- Warmup vs post-warmup metrics:
- Exit behavior transition:
- Notes:

## Method 18: Hybrid S-GRPO/GRPO Branching

Scripts:
- scripts/sgrpo/train_tiny_zero_hybrid_sgrpo.job

Script-level additions:
- algorithm.hybrid_branch.enable=True
- algorithm.hybrid_branch.correct_threshold=0.5
- algorithm.hybrid_branch.incorrect_extra_rollouts=3
- algorithm.hybrid_branch.tag_key=branch_mode
- plus S-GRPO settings and warmup

What code does:
- _build_hybrid_branch_rollouts in ray_trainer routes first-pass correct prompts to S-GRPO serial exits and incorrect prompts to GRPO extra full rollouts.
- compute_hybrid_branch_advantages computes S-GRPO and GRPO advantages on branch-specific subsets and merges them.
- Branch identity is stored in non_tensor_batch[tag_key].

Behavioral interpretation:
- Allocates compute adaptively: exploit serial exits for likely-correct prompts, spend extra samples on harder prompts.

Results (to fill later):
- Run IDs:
- first_pass_correct_frac:
- branch sample fractions and extra generations:
- Accuracy and token cost vs pure S-GRPO:
- Notes:

## Method 19: DAPO Stack (custom entrypoint)

Scripts:
- scripts/train_dapo_test.job

Script-level additions:
- Uses python3 -m recipe.dapo.main_dapo instead of main_ppo
- reward_model.reward_manager=dapo
- reward_model.overlong_buffer.* enabled
- algorithm.filter_groups.enable=True metric=acc
- actor clip asymmetry clip_ratio_low=0.2 clip_ratio_high=0.28
- actor loss_agg_mode=token-mean

What code does:
- recipe/dapo/main_dapo.py launches RayDAPOTrainer.
- recipe/dapo/dapo_ray_trainer.py applies group filtering: groups with zero metric variance are dropped.
- verl/workers/reward_manager/dapo.py applies base score and optional overlong penalty near max length.
- PPO policy loss still comes through standard actor loss plumbing with configured clip bounds.

Behavioral interpretation:
- Adds reward shaping against overlong responses and data-efficiency filtering on homogeneous groups.
- Tests a DAPO objective surface while still using GRPO-style advantage estimator in this launcher.

Results (to fill later):
- Run IDs:
- Fraction of groups filtered:
- Overlong penalty incidence:
- Accuracy and response length impact:
- Notes:

## Method 20: Entropy Logging Instrumentation

Scripts with explicit setting:
- Most test jobs set agent.entropy_logging.enable=False

What code does when enabled:
- ray_trainer preserves old_entropy from old policy log-prob computation.
- metric_utils.save_entropy_data writes entropy_step_*.pt containing old_entropy, attention_mask, rewards, uids.

Behavioral interpretation:
- Not a training method by itself, but important for analysis workflows.

Results (to fill later):
- Runs where enabled:
- Output directory used:
- Analysis summary:
- Notes:

## Script-by-Script Ledger (Specific Additions)

- scripts/scripts2/train_tiny_zero_adaptive.job: Wrapper that exports env defaults and delegates to adaptive shell launcher.
- scripts/scripts2/train_tiny_zero_adaptive.sh: Dynamic adaptive window (basic mode style knobs), 4K cap, no in-reward KL.
- scripts/scripts2/train_tiny_zero_cosine.sh: External cosine phase scheduler over max_response_length.
- scripts/scripts2/train_tiny_zero_full.job: Wrapper for fixed full-rollout baseline launcher.
- scripts/scripts2/train_tiny_zero_phased_schedule.sh: External sequential phase schedule with fixed lengths per run segment.
- scripts/scripts2/train_tiny_zero_phased.sh: External one-phase fixed length selected by shell arg.
- scripts/scripts2/train_tiny_zero.sh: Fixed 4096 baseline without adaptive controller.
- scripts/scripts2/train_tiny_zero_test.job: Small 2-GPU phased adaptive smoke-style launcher.
- scripts/scripts2/train_tiny_zero_vanilla.job: Wrapper for fixed 1024 vanilla launcher.
- scripts/scripts2/train_tiny_zero_vanilla.sh: Fixed 1024 with adaptive window explicitly disabled.
- scripts/sgrpo/train_tiny_zero_hybrid_sgrpo.job: Hybrid branch routing (S-GRPO for correct first-pass, GRPO extras for incorrect).
- scripts/sgrpo/train_tiny_zero_sgrpo.job: Pure S-GRPO serial exits with decayed rewards.
- scripts/sgrpo/train_tiny_zero_sgrpo_warmup.job: GRPO warmup before S-GRPO activation plus phased window.
- scripts/train_dapo_test.job: DAPO entrypoint, DAPO reward manager, overlong buffer penalties, filter-groups enabled.
- scripts/train_gamma_additive_tau_test.job: Discounted reasoning gamma + plain GRPO additive normalization tau sweep + incorrect penalty.
- scripts/train_gamma_test.job: Discounted reasoning gamma without additive normalization.
- scripts/train_grpo_lambda_flat_incorrect_trace_test.job: GRPO-lambda with flat incorrect trace variant.
- scripts/train_grpo_lambda_group_shortest_trace_test.job: GRPO-lambda with group-shortest shared trace base.
- scripts/train_grpo_lambda_second_trace_test.job: GRPO-lambda token normalization plus second trace weighting.
- scripts/train_grpo_lambda_test.job: Baseline GRPO-lambda test/eval launcher (variant disabled, val_only).
- scripts/train_grpo_lambda_token_normalization_additive_tau_test.job: Token normalization with additive tau in denominator.
- scripts/train_grpo_lambda_token_normalization_test.job: Token-normalized GRPO-lambda variant baseline.
- scripts/train_lambda_gamma.job: GRPO-lambda sequence gamma discount and reasoning-only trace mode.
- scripts/train_lead_grpo_lambda_test.job: LEAD + incorrect penalty on GRPO-lambda stack, fixed 4K window.
- scripts/train_lead_grpo_test.job: LEAD + incorrect penalty on GRPO (non-lambda) stack.
- scripts/train_normalphased_test.job: Resumed phased GRPO baseline in val_only mode.

## Open Execution Notes and Hygiene Checks

- scripts/scripts2 wrappers appear to reference scripts/train_tiny_zero*.sh while current files are under scripts/scripts2/.
- scripts/scripts2/train_tiny_zero_adaptive.sh sets algorithm.cosine_reward.enable=False, but cosine_reward does not appear in current ppo_trainer algorithm schema.
- scripts/train_lead_grpo_test.job includes trainer.resume_from_path=auto\ and trainer.total_training_steps=0, which should be sanity-checked before production runs.
- Several launchers are val_only or very short-step tests; classify results accordingly when comparing methods.

## Master Results Board (Fill Later)

- Baseline anchor run:
- Best GRPO-lambda variant so far:
- Best stability profile:
- Best token-efficiency profile:
- Most promising next experiment:
- Kill list (methods to stop pursuing):
