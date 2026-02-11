#!/usr/bin/env bash
# Run evaluation for a TinyZero-style checkpoint using the same pipeline
# (RayPPOTrainer._validate) as training, but without any training steps.
#
# Usage:
#   DATA_DIR=/path/to/data BASE_MODEL=/path/to/model \
#   EXPERIMENT_NAME=my_eval_run bash scripts/eval_tiny_zero.sh
#
# This will:
#   - load the tokenizer/model from BASE_MODEL
#   - load val data from $DATA_DIR/test.parquet
#   - run a single validation pass (no training)
#   - log val/test_score/* and val/acc/* to W&B under EXPERIMENT_NAME.

# set -euo pipefail

export DATA_DIR=${DATA_DIR:-./}
export BASE_MODEL=${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
export N_GPUS=${N_GPUS:-1}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-tinyzero_eval}

python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=1312 \
  data.max_prompt_length=256 \
  data.max_response_length=4096 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=72 \
  actor_rollout_ref.actor.ppo_micro_batch_size=12 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=12 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  critic.ppo_mini_batch_size=72 \
  critic.ppo_micro_batch_size=12 \
  trainer.logger=['wandb'] \
  trainer.val_before_train=True \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=TinyZero \
  trainer.experiment_name=${EXPERIMENT_NAME}_eval \
  trainer.total_epochs=1 2>&1 | tee verl_eval.log


