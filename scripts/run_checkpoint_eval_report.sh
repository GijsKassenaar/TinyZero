#!/usr/bin/env bash
set -euo pipefail

# Wrapper for scripts/eval_checkpoint_report.py
#
# Defaults evaluate the built-in checkpoint set sequentially via
# trainer val-only validation, using a 20-row subset from the test split.
#
# Example:
#   DATA_DIR=. SPLIT=test SAMPLE_COUNT=20 OUTPUT_DIR=reports/checkpoint_eval \
#   bash scripts/run_checkpoint_eval_report.sh
#
# Example with custom checkpoint list:
#   bash scripts/run_checkpoint_eval_report.sh \
#     --checkpoints_file scripts/my_checkpoints.txt

PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_DIR=${DATA_DIR:-.}
SPLIT=${SPLIT:-test}
SAMPLE_COUNT=${SAMPLE_COUNT:-20}
SAMPLE_MODE=${SAMPLE_MODE:-random}
SAMPLE_SEED=${SAMPLE_SEED:-42}
BASE_MODEL=${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
N_GPUS=${N_GPUS:-4}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
LOGGER=${LOGGER:-console}
PROJECT_NAME=${PROJECT_NAME:-TinyZero}
OUTPUT_DIR=${OUTPUT_DIR:-reports/checkpoint_eval}
RUN_NAME=${RUN_NAME:-}

CMD=(
  "$PYTHON_BIN" scripts/eval_checkpoint_report.py
  --data_dir "$DATA_DIR"
  --split "$SPLIT"
  --sample_count "$SAMPLE_COUNT"
  --sample_mode "$SAMPLE_MODE"
  --sample_seed "$SAMPLE_SEED"
  --base_model "$BASE_MODEL"
  --n_gpus "$N_GPUS"
  --rollout_tp_size "$ROLLOUT_TP_SIZE"
  --logger "$LOGGER"
  --project_name "$PROJECT_NAME"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi

# Forward any extra flags, for example:
# --checkpoints ...
# --checkpoints_file ...
CMD+=("$@")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
