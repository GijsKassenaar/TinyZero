# TinyZero

TinyZero is a reproduction and extension of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) on the **Countdown** (arithmetic expression search) and multiplication tasks, built on top of [veRL](https://github.com/volcengine/verl).

This fork adds a **compute‑efficient adaptive rollout curriculum** for RL with long‑context reasoning models, plus tooling to run ablations, log rich diagnostics, and evaluate trained checkpoints reproducibly.

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
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

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
