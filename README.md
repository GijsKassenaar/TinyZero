# TinyZero

TinyZero is a reproduction and extension of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) on the **Countdown** (arithmetic expression search) and multiplication tasks, built on top of [veRL](https://github.com/volcengine/verl).

On top of the original setup, this codebase adds a **compute‑efficient adaptive rollout curriculum** for RL with long‑context reasoning models, plus tooling to run ablations, log rich diagnostics, and evaluate trained checkpoints reproducibly.

> Key result (Countdown): a 1.5B DeepSeek‑distilled model trained with our **adaptive rollout** schedule matches and slightly exceeds a full 4K rollout baseline and even outperforms a 7B base model, while using tokens more efficiently.

---

## Installation

We recommend using a dedicated Conda environment.

```bash
conda create -n zero python=3.9
conda activate zero

# PyTorch (or skip and let vLLM install a matching wheel)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# vLLM + Ray
pip install "vllm==0.6.3" ray

# Install TinyZero (this repo) + veRL
pip install -e .

# FlashAttention 2 (optional but recommended for speed on H100/A100)
pip install flash-attn --no-build-isolation

# Quality of life
pip install wandb IPython matplotlib

Or on Snellius

conda activate zero ( directly activates right conda env) 
```




## Countdown task

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training
```
conda activate zero
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script, and checkout the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643)

**Single GPU**


Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with QWen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir=./
```

**Training on Adaptive Rollout**
```
export N_GPUS=4
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export DATA_DIR=./
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-adaptive
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_adaptive.sh
```

**Training on Fixed Rollout**
```
export N_GPUS=4
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export DATA_DIR=./
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-adaptive
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Adaptive rollout extensions

### Training modes

- **Full rollout (baseline)**  
  - Script: `scripts/train_tiny_zero.sh`  
  - Fixed maximum response length (e.g. 4096 tokens) for the entire run.

- **Adaptive window (ARPO‑style)**  
  - Script: `scripts/train_tiny_zero_adaptive.sh`  
  - Uses `AdaptiveSuccessWindowController` (`verl/trainer/ppo/adaptive_window.py`) to:
    - Start from a warm‑up window based on the mean length of correct responses.
    - Track batch‑level success rate and success length.
    - **Shrink** the window when success is high and the model under‑uses length.
    - **Grow** the window when success is low or the model is saturating the current window.
    - Occasionally explore the max window via epsilon‑greedy.
  - All behavior is controlled by `agent.adaptive_window.*` in `verl/trainer/config/ppo_trainer.yaml`.

- **Vanilla fixed‑length baseline**  
  - Script: `scripts/train_tiny_zero_vanilla.sh`  
  - Disables the adaptive controller (`agent.adaptive_window.enable=False`) and keeps a fixed `data.max_response_length`.

All these modes share the same PPO/GRPO trainer (`verl/trainer/ppo/ray_trainer.py`) and configuration file (`verl/trainer/config/ppo_trainer.yaml`).

### GRPO and reward shaping

- By default, this codebase uses **GRPO**:
  - `algorithm.adv_estimator=grpo`,
  - multiple samples per prompt (`actor_rollout_ref.rollout.n`),
  - actor‑side KL loss (`use_kl_loss=True`, `kl_loss_type=low_var_kl`).
- The Countdown reward is purely **binary correctness**:
  - `verl/utils/reward_score/countdown.py` uses `format_score=0.0`, so “format‑correct but wrong answer” gets 0.
  - This avoids reward hacking where the model gets partial credit for nicely formatted but incorrect equations.
- Optional **cosine reward shaping** can be enabled via `algorithm.cosine_reward.enable=True` to modulate rewards as a function of response length.

### Metrics and analysis

To study stability and efficiency, the trainer logs additional metrics to W&B:

- **Adaptive window behavior**
  - `adaptive_window/current_window`, `adaptive_window/mean_success_length`, `adaptive_window/success_rate`, etc.
- **Importance sampling diagnostics**
  - `actor/is_ratio/mean`, `actor/is_ratio/std`, `actor/is_ratio/p95`,
  - `actor/is_log_ratio/mean`, `actor/is_log_ratio/std`,
  - `actor/is_clip/outside_frac` (fraction of ratios beyond the PPO clip range).
- **Token usage**
  - `tokens/prompt_total`, `tokens/response_total`, `tokens/overall_total`.
- **Completion quality**
  - `completion/truncated_frac`, `completion/finished_frac`,
  - accuracy conditioned on truncation status,
  - average length of correct responses.
- **Difficulty‑conditioned accuracy**
  - `difficulty/3_acc`, `difficulty/4_acc` for Countdown, using the `difficulty` field added in preprocessing.
- **Learning curves**
  - `train/cumulative_reward` and `time/elapsed_s` to plot reward vs. wall‑clock.

These metrics make it easy to compare full‑rollout vs. adaptive vs. vanilla fixed‑length runs on both **final accuracy** and **compute efficiency (tokens / time)**.

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
