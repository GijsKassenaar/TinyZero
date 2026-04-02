# Checkpoint Validation Report

- Run: checkpoint_eval_20260401_153904
- Created: 2026-04-01T15:39:04
- Validation mode: trainer val-only (same pipeline as training _validate)
- Split: test
- Requested sample count: 20
- Actual sample count: 20
- Sample mode: random
- Sample seed: 42
- Length stats source: tokenizer:deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Subset parquet: reports/checkpoint_eval/checkpoint_eval_20260401_153904/subsets/test_random_20.parquet

| Name | Profile | Mean Reward | Accuracy | RespLen-C | RespLen-I | ReasonLen-C | ReasonLen-I | Records | Status | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| test_normalphased | normal_grpo | 0.8000 | 0.8000 | 342.2500 | 2048.0000 | 322.1875 | 0.0000 | 20 | ok | 261.0207 |
| test_discounted_reasoning_0.999 | discounted_reasoning | 0.7500 | 0.7500 | 114.4000 | 2048.0000 | 69.4000 | 0.0000 | 20 | ok | 131.7527 |
| test_grpo_lambda_0.99 | grpo_lambda_base | 0.8000 | 0.8000 | 272.5000 | 2048.0000 | 253.0625 | 0.0000 | 20 | ok | 139.0837 |
| test_grpo_lambda_sequence_gamma_discount | grpo_lambda_sequence_gamma_discount | 0.7000 | 0.7000 | 214.2143 | 1456.0000 | 194.9286 | 84.3333 | 20 | ok | 133.0380 |
| test_grpo_lambda_0.999_token_normalization | grpo_lambda_token_normalization | 0.8000 | 0.8000 | 106.1875 | 2048.0000 | 87.0000 | 0.0000 | 20 | ok | 132.9533 |
| test_grpo_lambda_second_trace | grpo_lambda_second_trace | 0.6000 | 0.6000 | 77.6667 | 1798.8750 | 61.0833 | 4.5000 | 20 | ok | 153.5547 |

## Artifacts

- Summary CSV: reports/checkpoint_eval/checkpoint_eval_20260401_153904/summary.csv
- Summary JSON: reports/checkpoint_eval/checkpoint_eval_20260401_153904/summary.json
- Logs directory: reports/checkpoint_eval/checkpoint_eval_20260401_153904/logs
- Validation dumps: reports/checkpoint_eval/checkpoint_eval_20260401_153904/validation_data
- Readable dumps: <validation_data>/<model>/75.readable.jsonl
