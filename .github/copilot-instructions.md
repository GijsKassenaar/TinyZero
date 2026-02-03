# TinyZero AI Agent Instructions

## Project Overview

TinyZero extends DeepSeek R1 Zero for the **Countdown** arithmetic reasoning task using **veRL** (FSDP PPO/GRPO). The key contribution is an **adaptive rollout curriculum** that dynamically adjusts response length based on success rate, achieving compute efficiency comparable to full 4K rollout.

**Core task**: Train models to generate arithmetic equations reaching a target number from available operands.

## Architecture & Data Flow

### Training Pipeline (PPO/GRPO)
- **Entry**: `verl/trainer/main_ppo.py` → `verl/trainer/ppo/ray_trainer.py`
- **Workers**: Ray-based distributed setup with colocated Actor/Rollout/Ref workers
- **Rollout**: vLLM backend (`actor_rollout_ref.rollout.*` in config)
- **Reward**: Binary correctness (0 or 1) from `verl/utils/reward_score/countdown.py`
  - Extracts `<answer>equation</answer>` tag
  - Validates equation uses available numbers correctly
  - **Critical**: `format_score=0.0` prevents reward hacking (no partial credit)

### Adaptive Window Controller
**File**: `verl/trainer/ppo/adaptive_window.py`

The controller tracks batch success rate and dynamically adjusts `data.max_response_length`:
- **Shrink**: High success + models under-use current window → reduce tokens
- **Grow**: Low success or models saturate window → increase capacity
- **Epsilon-greedy**: Occasional exploration at max window to prevent premature convergence

**Key metrics** (logged to W&B):
```
adaptive_window/current_window
adaptive_window/success_rate_ema
adaptive_window/mean_success_length
```

## Environment Setup

### Snellius HPC (Primary)
```bash
conda activate zero  # Pre-configured environment
```

### Local Development
```bash
conda create -n zero python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "vllm==0.6.3" ray flash-attn --no-build-isolation
pip install -e .  # Install TinyZero + veRL
```

## Critical Workflows

### 1. Data Preprocessing
```bash
python examples/data_preprocess/countdown.py --local_dir ./data
```
Output: `train.parquet`, `test.parquet` with columns:
- `prompt`: Chat format with target/numbers
- `reward_model`: Dict with `ground_truth` (target, numbers)
- `difficulty`: Optional field for stratified analysis

### 2. Training Modes

**Adaptive (recommended)**:
```bash
export N_GPUS=4
export ROLLOUT_TP_SIZE=2  # Must divide N_GPUS
bash scripts/train_tiny_zero_adaptive.sh
```

**Fixed full rollout (baseline)**:
```bash
bash scripts/train_tiny_zero.sh
```

**Vanilla fixed-length**:
```bash
bash scripts/train_tiny_zero_vanilla.sh  # agent.adaptive_window.enable=False
```

**Key config overrides** (all scripts use `verl/trainer/config/ppo_trainer.yaml`):
- `data.max_response_length`: Upper bound (e.g., 4096)
- `agent.adaptive_window.*`: Controller hyperparameters
- `algorithm.adv_estimator=grpo`: Group relative advantage
- `actor_rollout_ref.actor.use_kl_loss=True`: Actor-side KL penalty
- `trainer.project_name` / `trainer.experiment_name`: W&B logging

### 3. Slurm Job Management (Snellius)
```bash
sbatch scripts/train_tiny_zero_entropy_test.job
squeue -u $USER
tail -f slurm-<JOB_ID>.out
scancel <JOB_ID>
```

### 4. Downloading Results
```bash
scp -r gkassenaar@snellius.surf.nl:/home/gkassenaar/TinyZero/checkpoints/*/entropy_data ./outputs_Gijs/
```

### 5. Evaluation
```bash
python scripts/eval_countdown_model.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_dir ./data \
  --split test \
  --max_response_length 4096 \
  --batch_size 128
```

Outputs mean reward and strict accuracy (reward >= 0.99).

## Entropy Analysis Tools

**Purpose**: Compare token-level entropy/varentropy between correct and incorrect responses.

### GRPO Group Tracking

**Critical finding**: GRPO groups (multiple responses to the same prompt) are:
1. **Initially stored together** with `interleave=True` during batch creation
2. **Mixed up by `_balance_batch`** for load balancing across GPUs (see warning in code)
3. **Tracked via UIDs** in `batch.non_tensor_batch['uid']` for advantage computation
4. **UIDs are saved** in entropy data (as of latest code) to reconstruct groups

**Group reconstruction**:
- Each unique UID represents one prompt
- Responses with the same UID form a GRPO group
- Default: 4 responses per prompt (`actor_rollout_ref.rollout.n=4`)
- `compute_grpo_outcome_advantage` uses UIDs to re-group and normalize advantages within each prompt group

### Batch Analysis (Main)
```bash
python entropy_analysis_batch.py outputs_Gijs/entropy_analysis_1024-2048/entropy_data
```

**Key features**:
- Auto-detects actual response lengths (handles variable 1024→2048 transitions)
- Groups responses by prompt (4 samples/prompt) for pattern analysis
- Generates positional plots showing entropy evolution by token position
- **Critical fix**: Uses `correct_max_valid_pos` / `incorrect_max_valid_pos` to avoid zero-padding artifacts

**Outputs** (saved to `entropy_analysis/`):
1. `success_rate_over_time.png`
2. `entropy_varentropy_comparison.png`
3. `positional_comparison.png` (averaged across steps)
4. `positional_evolution_grid.png` (6 key steps side-by-side)
5. `per_step_positional/` (individual plots per step)

### Group Pattern Analysis
Automatically analyzes success distribution within prompt groups:
```
0/4 correct: 12.5%
1/4 correct: 18.2%
...
4/4 correct: 45.3%
```

## Project-Specific Conventions

### Configuration System
- **Base**: `verl/trainer/config/ppo_trainer.yaml`
- **Override**: Command-line args in shell scripts (OmegaConf syntax)
- **Never edit** the base YAML for experiments—always use CLI overrides

### Reward Function Contract
All reward scorers must accept:
```python
def compute_score(solution_str: str, ground_truth: dict, 
                 method='strict', format_score=0.0, score=1.0) -> float
```

For Countdown:
- `ground_truth = {'target': int, 'numbers': List[int]}`
- Returns 0.0 or 1.0 (no partial credit)

### Tensor Parallel Size
`ROLLOUT_TP_SIZE` must evenly divide `N_GPUS`. Common configs:
- 2 GPUs: TP=1 or 2
- 4 GPUs: TP=1, 2, or 4
- Invalid: 3 GPUs with TP=2

### W&B Metrics Hierarchy
```
adaptive_window/*  # Controller state
actor/*            # Policy gradients, IS ratios
tokens/*           # Token usage (prompt/response/total)
completion/*       # Truncation, finish rates
difficulty/*       # Task-specific accuracy by difficulty
train/*            # Cumulative reward, loss curves
```

## Common Pitfalls

1. **OOM with large models (3B+)**: Enable `critic.model.enable_gradient_checkpointing=True`
2. **Zero-padding in plots**: Ensure entropy analysis uses `*_max_valid_pos` to truncate plots at actual response length
3. **Reward hacking**: Never set `format_score > 0` for Countdown (encourages malformed but "correct-looking" answers)
4. **TP size mismatch**: Verify `ROLLOUT_TP_SIZE` divides `N_GPUS` evenly
5. **vLLM backend**: Use `VLLM_ATTENTION_BACKEND=XFORMERS` on H100/A100

## Key Files Reference

- **Main trainer**: `verl/trainer/ppo/ray_trainer.py` (1046 lines, handles worker orchestration)
- **Adaptive controller**: `verl/trainer/ppo/adaptive_window.py`
- **Reward function**: `verl/utils/reward_score/countdown.py` (extract/validate/score equations)
- **Data prep**: `examples/data_preprocess/countdown.py`
- **Config**: `verl/trainer/config/ppo_trainer.yaml`
- **Analysis**: `entropy_analysis_batch.py` (batch analysis with group patterns)

## Debugging Workflows

1. **Check adaptive window behavior**: Monitor W&B metrics `adaptive_window/current_window` vs `adaptive_window/success_rate`
2. **Validate rewards**: Use `scripts/eval_countdown_single.py --index 0` to inspect single example
3. **Entropy artifacts**: Verify `correct_max_valid_pos` matches actual response lengths in saved `.pt` files
4. **Group patterns**: Check terminal output for `GROUP PATTERN ANALYSIS` section after training

## Master Thesis Context

This codebase extends TinyZero for Gijs's thesis on adaptive rollout curricula. Key additions:
- Entropy logging during PPO (`entropy_step_*.pt` files)
- Batch analysis comparing correct/incorrect reasoning traces
- Group pattern analysis (4 responses per prompt)
- Variable response length support (1024→2048 transitions)
