# TinyZero AI Agent Instructions

## Project Overview

TinyZero extends DeepSeek R1 Zero for the **Countdown** arithmetic reasoning task using **veRL** (FSDP PPO/GRPO). The key contribution is a **compute-efficient adaptive rollout curriculum** that dynamically adjusts response length, matching full 4K rollout performance while using tokens more efficiently.

**Core task**: Train 1.5B/3B models to generate arithmetic equations reaching a target number from available operands.

## Architecture & Data Flow

### Training Pipeline
```
verl/trainer/main_ppo.py (Hydra entry)
 → TaskRunner (Ray remote actor)
   → creates workers: Actor+Rollout+Ref (colocated), Critic, RewardModel
   → creates RayPPOTrainer
     → trainer.fit() — main loop in verl/trainer/ppo/ray_trainer.py
```
- **Single-controller**: driver process coordinates workers via Ray RPC
- **DataProto**: universal container with `batch` (TensorDict) + `non_tensor_batch` + `meta_info`
- **`_balance_batch()`**: reorders data so each DP rank gets similar total tokens (breaks GRPO group locality — UIDs track group membership)
- **Rollout**: vLLM backend (`actor_rollout_ref.rollout.*`)
- **Reward**: Binary 0/1 from `verl/utils/reward_score/countdown.py` — extracts `<answer>equation</answer>`, validates numbers, evals equation. **`format_score=0.0`** prevents reward hacking.

### Custom Research Extensions

All follow the same integration pattern:
1. Dataclass config with `enable: bool = False` in a dedicated `verl/trainer/ppo/<feature>.py`
2. Controller class with `__init__(config, tokenizer, max_response_length)`
3. Config exposed as nested keys in `ppo_trainer.yaml` under `algorithm.*` or `agent.*`
4. Activated via CLI override: `algorithm.<feature>.enable=True`
5. W&B metrics namespaced as `<feature>/metric_name`

| Feature | Config prefix | Module | Purpose |
|---------|--------------|--------|---------|
| Adaptive window | `agent.adaptive_window.*` | `adaptive_window.py` | Dynamic max_tokens based on success rate |
| Truncation recovery | `algorithm.truncation_recovery.*` | `truncation_recovery.py` | Re-prompt truncated responses with answer inducer |
| S-GRPO | `algorithm.sgrpo.*` | `sgrpo.py` | Serial decaying-reward multi-exit generation |
| Entropy logging | `agent.entropy_logging.*` | (cluster-deployed) | Save per-token entropy `.pt` files |

### Adaptive Window Modes
- **`basic`** / **`ema`** / **`rolling`**: Track success lengths, shrink when success high, grow when low, epsilon-greedy exploration
- **`phased`**: Fixed step-based schedule, e.g. `'[[50,2048]]'` = switch to 2048 at step 50

### Advantage Estimators (`algorithm.adv_estimator`)
`gae`, `grpo`, `grpo_vectorized`, `grpo_passk`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `remax`, `rloo`, `rloo_vectorized`, `opo`, `gpg` — see `verl/trainer/ppo/core_algos.py`

### Policy Losses (registered in `core_algos.py`)
`vanilla`, `gspo`, `gpg`, `clip_cov`, `kl_cov`, `geo_mean`

## Configuration System

- **Base YAML**: `verl/trainer/config/ppo_trainer.yaml` (Hydra + OmegaConf)
- **Never edit** the base YAML for experiments — always use CLI overrides in scripts
- **Config schema**: `verl/base_config.py` — immutable dataclasses implementing `Mapping`
- **Key sections**: `algorithm.*`, `agent.*`, `actor_rollout_ref.*`, `trainer.*`, `data.*`

## Critical Workflows

### Data Preprocessing
```bash
python examples/data_preprocess/countdown.py --local_dir ./data
# Output: train.parquet, test.parquet (columns: prompt, reward_model, difficulty)
```

### Training Modes
Scripts are mixed: most current experiment launchers are self-contained `.job` files, while some older launchers still keep the `python3 -m verl.trainer.main_ppo` command in a companion `.sh`.

| Mode | Script | Key overrides |
|------|--------|---------------|
| Adaptive (recommended) | `train_tiny_zero_adaptive.sh` | `agent.adaptive_window.enable=True` |
| Full rollout baseline | `train_tiny_zero.sh` | Fixed `data.max_response_length=4096` |
| Vanilla fixed-length | `train_tiny_zero_vanilla.sh` | `agent.adaptive_window.enable=False` |
| Phased schedule | `train_tiny_zero_phased_schedule.sh` | `agent.adaptive_window.mode=phased` |
| S-GRPO | `train_tiny_zero_sgrpo.job` | `algorithm.sgrpo.enable=True` |

**Env vars** set before running: `N_GPUS`, `ROLLOUT_TP_SIZE` (must divide N_GPUS), `BASE_MODEL`, `DATA_DIR`, `EXPERIMENT_NAME`.

### Slurm (Snellius HPC)
```bash
conda activate zero
sbatch scripts/train_tiny_zero_<mode>.job
squeue -u $USER && tail -f slurm-<JOB_ID>.out
```

### Evaluation
```bash
python scripts/eval_countdown_model.py --model <path> --data_dir ./data --split test --max_response_length 4096
python scripts/eval_countdown_single.py --index 0  # Inspect single example
```

### Entropy Analysis (offline)
```bash
# Download from cluster
scp -r gkassenaar@snellius.surf.nl:/home/gkassenaar/TinyZero/checkpoints/*/entropy_data ./outputs_Gijs/
# Analyze
python entropy_analysis_batch.py outputs_Gijs/entropy_analysis_1024-2048/entropy_data
```
Reads `entropy_step_*.pt` files → generates plots in `entropy_analysis/`. Uses `correct_max_valid_pos` / `incorrect_max_valid_pos` to avoid zero-padding artifacts.

## Project-Specific Conventions

### Reward Function Contract
```python
def compute_score(solution_str: str, ground_truth: dict,
                 method='strict', format_score=0.0, score=1.0) -> float
# ground_truth = {'target': int, 'numbers': List[int]}
# Returns 0.0 or 1.0 — no partial credit for Countdown
```

### GRPO Group Tracking
- Groups created with `interleave=True`, then **shuffled by `_balance_batch()`**
- UIDs in `batch.non_tensor_batch['uid']` track group membership
- `compute_grpo_outcome_advantage` uses UIDs to re-group for within-group normalization
- Default: 4 responses/prompt (`actor_rollout_ref.rollout.n=4`)

### Adding a New Training Mode
1. Create `scripts/train_tiny_zero_<mode>.sh` with CLI overrides
2. Create `scripts/train_tiny_zero_<mode>.job` with SLURM headers + env vars
3. Add feature config (if needed) as `@dataclass` in `verl/trainer/ppo/<feature>.py`
4. Wire into `ray_trainer.py`'s `fit()` loop; namespace W&B metrics as `<feature>/*`

### W&B Metrics
```
adaptive_window/*           # Window size, success rate, success length stats
truncation_recovery/*       # Truncation counts, recovery rate
actor/*                     # Policy loss, entropy, IS ratios, grad norm, lr
critic/*                    # KL, scores, rewards, advantages, values
tokens/*                    # Prompt/response/overall totals
completion/*                # Truncation/finish rates, group_XofN_correct_pct
difficulty/*                # Per-difficulty accuracy
train/*                     # Cumulative reward
```
Full reference: `WANDB_METRICS.md` in repo root.

## Common Pitfalls

1. **OOM (3B+ models)**: Add `critic.model.enable_gradient_checkpointing=True`
2. **TP mismatch**: `ROLLOUT_TP_SIZE` must evenly divide `N_GPUS`
3. **Reward hacking**: Never set `format_score > 0` for Countdown
4. **vLLM backend**: Use `VLLM_ATTENTION_BACKEND=XFORMERS` on H100/A100
5. **Entropy plots**: Always use `*_max_valid_pos` to truncate at actual response length
6. **Rollout IS weights**: `algorithm.rollout_is_threshold` (null = disabled) — set carefully to avoid clipping too aggressively

## Key Files

| File | Purpose |
|------|---------|
| `verl/trainer/ppo/ray_trainer.py` | Main training loop, worker orchestration |
| `verl/trainer/ppo/adaptive_window.py` | Dynamic response length controller |
| `verl/trainer/ppo/truncation_recovery.py` | Re-prompt truncated samples |
| `verl/trainer/ppo/sgrpo.py` | Serial multi-exit GRPO |
| `verl/trainer/ppo/core_algos.py` | Advantage estimators, policy losses |
| `verl/trainer/ppo/metric_utils.py` | Metric computation helpers |
| `verl/trainer/config/ppo_trainer.yaml` | Base Hydra config |
| `verl/utils/reward_score/countdown.py` | Binary reward function |
| `examples/data_preprocess/countdown.py` | Data preparation |
| `entropy_analysis_batch.py` | Offline entropy analysis with group patterns |
