# S-GRPO Implementation Report

## Purpose

This document explains how S-GRPO is implemented in the current TinyZero codebase, what changed to make it work with the current veRL stack, and how the new warmup behavior works.

S-GRPO stands for Serial-Group Decaying-Reward Policy Optimization. The main idea is:

1. Generate one full reasoning trajectory per prompt.
2. Cut that trajectory at several intermediate exit points.
3. Continue from those cut points with short answer-oriented follow-up generations.
4. Score all exits together as a prompt-local group.
5. Prefer earlier correct exits by decaying reward as exit order gets later.

In this repo, S-GRPO is implemented as a two-phase rollout strategy inside the PPO trainer.

## High-Level Flow

For each prompt, the active S-GRPO path does this:

1. Generate one full response.
2. Choose `num_exits` exit positions across the full response.
3. For exits `1..K-1`, build a new prompt consisting of:
   - original prompt
   - truncated partial reasoning
   - an answer inducer string
4. Run a second generation pass from those truncated prompts.
5. Reconstruct a serial group of size `K`:
   - exits `1..K-1`: `partial + inducer + continuation`
   - exit `K`: original full response
6. Attach `exit_order` to each member of the group.
7. Compute binary Countdown reward as usual.
8. Convert reward to S-GRPO advantage by decaying later exits and normalizing within each prompt group.

## Main Files and Their Roles

### `verl/trainer/ppo/sgrpo.py`

This is the algorithm controller. It owns:

- exit placement
- truncated continuation prompt construction
- recombination of all exits into one serial group
- S-GRPO-specific metrics

### `verl/trainer/ppo/core_algos.py`

This registers the S-GRPO advantage estimator and computes the decayed, group-normalized outcome advantage.

### `verl/trainer/ppo/ray_trainer.py`

This is where S-GRPO is wired into the actual PPO training loop.

It decides:

- whether S-GRPO is enabled at all
- whether warmup is still active
- when to perform the second continuation pass
- how many copies of the batch to create
- when to attach `exit_order`
- when to log S-GRPO-specific metrics

### `verl/experimental/agent_loop/agent_loop.py`

This file was minimally extended so S-GRPO continuation prompts can run through the async rollout stack without pretending to be normal dataset prompts.

### `verl/trainer/config/algorithm.py`

This defines the typed `SGRPOConfig` used by the trainer.

### `verl/trainer/config/ppo_trainer.yaml`

This exposes the default config fields under `algorithm.sgrpo`.

### `scripts/train_tiny_zero_sgrpo.job`

Self-contained S-GRPO launcher without warmup.

### `scripts/train_tiny_zero_sgrpo_warmup.job`

Self-contained S-GRPO launcher with configurable warmup, defaulting to 30 steps.

## Detailed Breakdown by Function

### 1. Config Definition

File: `verl/trainer/config/algorithm.py`

`SGRPOConfig` defines the algorithm parameters:

- `enable`: turns S-GRPO on
- `warmup_steps`: number of PPO steps before S-GRPO serial exits activate
- `num_exits`: total exits per prompt, including the full response
- `decay_factor`: reward decay base for later exits
- `exit_method`: currently `uniform`
- `answer_inducer`: string appended before continuation generation

File: `verl/trainer/config/ppo_trainer.yaml`

The same fields are exposed under `algorithm.sgrpo`, so they can be overridden from job scripts or the CLI.

## 2. Trainer Initialization

File: `verl/trainer/ppo/ray_trainer.py`

When the trainer starts, it reads `algorithm.sgrpo` and constructs `SGRPOController` if S-GRPO is enabled.

Important constraint:

- once S-GRPO is active, the trainer uses one base rollout internally and then expands it into serial exits

That means the normal `actor_rollout_ref.rollout.n` setting is used for warmup GRPO behavior, not for the active S-GRPO phase.

## 3. Exit Placement

File: `verl/trainer/ppo/sgrpo.py`

Function: `get_uniform_exit_positions(response, num_exits, eos_token_id)`

This function chooses where the response is cut.

Behavior:

1. Find the effective response length up to EOS.
2. Divide that length uniformly into `num_exits` positions.
3. Return one exit position per exit order.

Example with `num_exits = 4`:

- exit 1 at about 25%
- exit 2 at about 50%
- exit 3 at about 75%
- exit 4 at 100%

The last exit is the full response.

## 4. Phase 1: Full Generation

File: `verl/trainer/ppo/ray_trainer.py`

Inside the main training loop, the trainer first performs the normal rollout call.

If S-GRPO is inactive, that is the only generation pass.

If S-GRPO is active, that full response becomes the source trajectory for the second pass.

## 5. Phase 2: Truncated Continuation Prompt Construction

File: `verl/trainer/ppo/sgrpo.py`

Function: `_prepare_truncated_prompts(full_responses)`

This function converts each full response into `K-1` continuation prompts.

For each original sample:

1. Read the original prompt.
2. Read the generated full response.
3. Compute exit positions.
4. For each early exit:
   - keep the prefix of the response up to the exit point
   - append the answer inducer tokens
   - create a new tokenized prompt for continuation generation

The answer inducer currently is:

`Time is limited, stop thinking and start answering.\n</think>\n<answer>`

Functionally, it forces the continuation pass to stop extending the chain-of-thought and instead produce a final answer.

This function also:

- left-pads continuation prompts to equal length
- constructs `input_ids`, `attention_mask`, and `position_ids`
- marks the batch with `meta_info['prefilled_prompt_mode'] = True`
- marks the batch with `non_tensor_batch['__skip_reward_compute__'] = True`

Those two flags are essential for async-stack compatibility.

## 6. Async Rollout Compatibility Layer

File: `verl/experimental/agent_loop/agent_loop.py`

### Why this was needed

The current async rollout stack expects normal dataset-style prompts with `raw_prompt`. S-GRPO continuation prompts are different: they are synthetic, already-tokenized prompt prefixes built from an earlier model response.

The old direct-worker workaround caused an event-loop error in the async environment. So the fix was to keep S-GRPO inside the async manager and add a narrow tokenized prompt path.

### What was added

#### `prefilled_prompt_mode`

During batch generation, the agent loop checks `batch.meta_info['prefilled_prompt_mode']`.

If enabled, each task receives:

- `prefilled_prompt_ids`
- `prefilled_attention_mask`
- `target_prompt_length`

#### `_run_prefilled_prompt(...)`

This function sends the already-tokenized prompt directly to the async server manager using `prompt_ids`.

That avoids the need for `raw_prompt` and avoids reconstructing a chat message tree.

#### `_agent_loop_postprocess(...)`

This was adjusted to:

- only attach `raw_prompt` if it actually exists
- pad prompts using `target_prompt_length` when provided

That ensures synthetic continuation prompts are postprocessed into the same tensor format as normal rollout outputs.

#### `_compute_score(...)`

This now exits early when `__skip_reward_compute__` is set.

That is important because these continuation generations are not standalone training samples. Reward should only be computed on the final recombined serial-group batch.

## 7. Serial Group Recombination

File: `verl/trainer/ppo/sgrpo.py`

Function: `_combine_serial_group(full_responses, truncated_continuations, exit_positions)`

This function builds the final S-GRPO training batch.

For each original prompt, it creates `num_exits` versions:

1. Exit `K`: the original full response
2. Exits `1..K-1`: `partial + inducer + continuation`

It then assembles:

- `input_ids`
- `responses`
- `attention_mask`
- `prompts`
- optional `position_ids`

It also propagates non-tensor metadata across all exits so things like `uid`, rule-based reward info, and other sample metadata survive the expansion.

Finally, it returns:

- `serial_data`: the expanded batch
- `exit_order`: a tensor with values `1..K`

That `exit_order` is what allows the trainer to apply exit-dependent reward decay later.

## 8. Advantage Computation

File: `verl/trainer/ppo/core_algos.py`

Function: `compute_sgrpo_outcome_advantage(...)`

This is the core S-GRPO scoring rule.

For each sample:

1. Sum token-level rewards into one sequence score.
2. Convert `exit_order` into a decay divisor:

   - exit 1: divide by `decay_factor^(0)`
   - exit 2: divide by `decay_factor^(1)`
   - exit 3: divide by `decay_factor^(2)`
   - etc.

3. Zero out incorrect samples.
4. Group samples by prompt `uid`.
5. Compute group mean and std across the serial exits.
6. Normalize within each prompt group.
7. Broadcast the normalized score across valid response tokens.

Effectively, earlier correct exits are more valuable than later correct exits, and the comparison is always local to the exits from the same prompt.

## 9. Trainer-Side S-GRPO Activation

File: `verl/trainer/ppo/ray_trainer.py`

### `_is_sgrpo_active()`

This helper implements the warmup gate:

- if S-GRPO is disabled, return `False`
- if S-GRPO is enabled, return `self.global_steps >= warmup_steps`

### Main fit loop behavior

When `sgrpo_active` is `False`:

1. The trainer repeats the batch using normal rollout logic.
2. It does one standard generation pass.
3. It does not create truncated prompts.
4. It does not attach `exit_order` from serial exits.
5. It still uses the S-GRPO advantage estimator, but with synthetic `exit_order = 1` for every sample.

When `sgrpo_active` is `True`:

1. The trainer does one full rollout per prompt.
2. It calls `create_serial_group_two_phase(...)`.
3. That triggers the continuation pass.
4. The trainer expands the original batch by `num_exits`.
5. It unions that expanded batch with the recombined serial-group outputs.
6. It attaches real `exit_order`.
7. It logs S-GRPO per-exit metrics.

The trainer also logs:

- `sgrpo/active`
- `sgrpo/warmup_steps`

## 10. Warmup Implementation

Warmup is not a separate training mode. It is a gate inside the normal S-GRPO trainer path.

### Config

The warmup parameter is:

`algorithm.sgrpo.warmup_steps`

### Exact behavior

If `warmup_steps = 50`:

- steps `0` through `49`: warmup mode
- step `50` onward: S-GRPO serial exits are active

### What happens during warmup

Warmup now behaves like normal GRPO even though the run is configured as an S-GRPO job overall.

The important distinction is:

- the warmup phase uses the normal `actor_rollout_ref.rollout.n`
- once S-GRPO becomes active, the trainer always uses one base rollout internally before creating serial exits

So warmup does **not** perform the second S-GRPO continuation pass yet.

Instead:

1. `actor_rollout_ref.rollout.n` normal rollouts are generated per prompt.
2. No serial-group expansion happens.
3. No real `exit_order` tensor exists.
4. The trainer switches the effective advantage estimator to GRPO for that step.

That means warmup behaves like ordinary GRPO before the S-GRPO phase begins:

- no exit decay penalty yet
- no second continuation pass yet
- group normalization happens across the normal `rollout.n` samples, as in GRPO

This design keeps the switch clean because the training loop, config, and optimizer path remain the same before and after activation.

### Why this warmup design is useful

It avoids a hard algorithm swap in the optimizer path. Only the rollout construction changes at the activation boundary.

So warmup is:

- easy to configure
- easy to log
- low-risk relative to the rest of the trainer
- compatible with the current async rollout stack

## What Changed to Make S-GRPO Work in the Current Codebase

The current implementation required several changes beyond the original `sgrpo.py` file.

### 1. Restored config wiring

S-GRPO config had to be reintroduced into the live typed config and base YAML.

### 2. Restored trainer wiring

The trainer now:

- instantiates `SGRPOController`
- calls it during generation
- attaches `exit_order`
- logs S-GRPO metrics

### 3. Registered the advantage estimator

The estimator `sgrpo` is now registered in `core_algos.py`.

### 4. Fixed config instantiation

The live trainer now constructs `SGRPOConfig` directly from the OmegaConf container instead of assuming a `_target_`-based dataclass instantiation path.

### 5. Made S-GRPO compatible with async rollout

The continuation pass no longer relies on the sync direct worker path in async mode.

Instead, it uses:

- `prefilled_prompt_mode`
- `_run_prefilled_prompt(...)`
- reward skipping for synthetic continuation subcalls

This was the main compatibility fix for the current veRL rollout stack.

### 6. Added warmup support

S-GRPO can now be delayed until an arbitrary training step using `algorithm.sgrpo.warmup_steps`.

## Current Launchers

### `scripts/train_tiny_zero_sgrpo.job`

Runs S-GRPO immediately.

### `scripts/train_tiny_zero_sgrpo_warmup.job`

Runs warmup first, then activates S-GRPO.

This launcher also enables adaptive window in `phased` mode with schedule:

- steps `0..49`: `max_tokens = 1024`
- step `50+`: `max_tokens = 2048`

By default it sets:

`SGRPO_WARMUP_STEPS=50`

and uses:

`actor_rollout_ref.rollout.n=4`

and

`ADAPTIVE_WINDOW_SCHEDULE=[[0,1024],[50,2048]]`

So you can run:

```bash
sbatch scripts/train_tiny_zero_sgrpo_warmup.job
```

Or change the warmup length:

```bash
SGRPO_WARMUP_STEPS=50 sbatch scripts/train_tiny_zero_sgrpo_warmup.job
```

## Practical Summary

The current S-GRPO implementation is best understood as:

1. one full rollout
2. multiple truncated continuation rollouts
3. recombination into a serial exit group
4. earlier-exit reward preference through decay
5. prompt-local group normalization for the advantage

Warmup delays steps 2 through 4 while keeping the rest of the training path consistent.

That means the system starts with normal GRPO-style multi-sample warmup (using configured `rollout.n`) and later upgrades into full S-GRPO without changing the overall training framework.