# Quick Analysis

- Run dir: /gpfs/home1/gkassenaar/TinyZero/reports/checkpoint_eval/checkpoint_eval_20260401_153904
- Baseline: test_normalphased
- Max response length threshold: 2048
- Generated: 2026-04-01T17:18:55

## Top Model

- Name: test_normalphased
- Mean reward: 0.8000
- Accuracy: 0.8000

## Model Comparison

| Model | Reward | Acc | dReward vs Base | dAcc vs Base | RespLen-C | RespLen-I | ReasonLen-C | ReasonLen-I | Trunc-I | MeanRespAll | MeanReasonAll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_normalphased | 0.8000 | 0.8000 | 0.0000 | 0.0000 | 342.2500 | 2048.0000 | 322.1875 | 0.0000 | 1.0000 | 683.4000 | 257.7500 |
| test_grpo_lambda_0.99 | 0.8000 | 0.8000 | 0.0000 | 0.0000 | 272.5000 | 2048.0000 | 253.0625 | 0.0000 | 1.0000 | 627.6000 | 202.4500 |
| test_grpo_lambda_0.999_token_normalization | 0.8000 | 0.8000 | 0.0000 | 0.0000 | 106.1875 | 2048.0000 | 87.0000 | 0.0000 | 1.0000 | 494.5500 | 69.6000 |
| test_discounted_reasoning_0.999 | 0.7500 | 0.7500 | -0.0500 | -0.0500 | 114.4000 | 2048.0000 | 69.4000 | 0.0000 | 1.0000 | 597.8000 | 52.0500 |
| test_grpo_lambda_sequence_gamma_discount | 0.7000 | 0.7000 | -0.1000 | -0.1000 | 214.2143 | 1456.0000 | 194.9286 | 84.3333 | 0.6667 | 586.7500 | 161.7500 |
| test_grpo_lambda_second_trace | 0.6000 | 0.6000 | -0.2000 | -0.2000 | 77.6667 | 1798.8750 | 61.0833 | 4.5000 | 0.8750 | 766.1500 | 38.4500 |

## Notes

- RespLen-C/I: mean response token length for correct/incorrect outputs.
- ReasonLen-C/I: mean reasoning token length (text before </think>) for correct/incorrect outputs.
- Trunc-I: fraction of incorrect outputs with response_len >= max_response_length.

- Source summary CSV: /gpfs/home1/gkassenaar/TinyZero/reports/checkpoint_eval/checkpoint_eval_20260401_153904/summary.csv
- Source summary JSON: /gpfs/home1/gkassenaar/TinyZero/reports/checkpoint_eval/checkpoint_eval_20260401_153904/summary.json
- Analysis JSON: /gpfs/home1/gkassenaar/TinyZero/reports/checkpoint_eval/checkpoint_eval_20260401_153904/quick_analysis.json
