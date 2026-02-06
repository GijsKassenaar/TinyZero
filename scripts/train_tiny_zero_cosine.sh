#!/usr/bin/env bash
# TinyZero training with a cosine-like scheduled increase of max response length.
# This is a simple discrete scheduler over several phases; each phase runs a
# separate training job with a different max_response_length chosen from a
# cosine ramp between L_MIN and L_MAX.
#
# Environment variables expected:
#   DATA_DIR          : directory containing train.parquet / test.parquet
#   BASE_MODEL        : HF model path
#   N_GPUS            : number of GPUs
#   ROLLOUT_TP_SIZE   : tensor model parallel size for rollout
#   EXPERIMENT_NAME   : base experiment name (optional; default: tinyzero_cosine)
#
# Optional overrides:
#   L_MIN             : minimum max_response_length (default: 512)
#   L_MAX             : maximum max_response_length (default: 2048)
#   N_PHASES          : number of cosine phases (default: 3)

set -euo pipefail

L_MIN=${L_MIN:-512}
L_MAX=${L_MAX:-4096}
N_PHASES=${N_PHASES:-3}

: "${DATA_DIR:?Must set DATA_DIR}"
: "${BASE_MODEL:?Must set BASE_MODEL}"
: "${N_GPUS:?Must set N_GPUS}"
: "${ROLLOUT_TP_SIZE:?Must set ROLLOUT_TP_SIZE}"
: "${EXPERIMENT_NAME:=tinyzero_cosine}"

if (( N_PHASES < 1 )); then
  echo "N_PHASES must be >= 1"
  exit 1
fi

echo "Running cosine schedule with L_MIN=${L_MIN}, L_MAX=${L_MAX}, N_PHASES=${N_PHASES}"

for (( PHASE=0; PHASE< N_PHASES; PHASE++ )); do
  # Compute phase-specific max_response_length using a cosine ramp:
  # L(t) = L_min + 0.5 * (L_max - L_min) * (1 - cos(pi * t / (N_PHASES-1)))
  MAX_RESP_LEN=$(python - "$PHASE" "$N_PHASES" "$L_MIN" "$L_MAX" << 'EOF'
import sys, math
phase = int(sys.argv[1])
n_phases = int(sys.argv[2])
L_min = int(sys.argv[3])
L_max = int(sys.argv[4])
if n_phases <= 1:
    L = L_max
else:
    T = n_phases - 1
    x = phase / T
    L = L_min + 0.5 * (L_max - L_min) * (1.0 - math.cos(math.pi * x))
print(int(round(L)))
EOF
)

  EXP_SUFFIX="cosine_p${PHASE}_len${MAX_RESP_LEN}"
  echo "Phase ${PHASE}/${N_PHASES-1}: max_response_length=${MAX_RESP_LEN}"

  python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=1312 \
  data.max_prompt_length=256 \
  data.max_response_length=$MAX_RESP_LEN \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=72 \
  actor_rollout_ref.actor.ppo_micro_batch_size=12 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=12 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.ppo_mini_batch_size=72 \
  critic.ppo_micro_batch_size=12 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['wandb'] \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=500 \
  trainer.test_freq=100 \
  trainer.project_name=TinyZero \
  trainer.experiment_name=${EXPERIMENT_NAME}_$EXP_SUFFIX \
  trainer.total_epochs=15 2>&1 | tee verl_demo_${EXP_SUFFIX}.log

done


