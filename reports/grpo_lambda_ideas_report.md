# GRPO-Lambda Ideas Summary

Date: 2026-04-01

This note briefly summarizes the current GRPO-lambda ideas in TinyZero and proposes additional ideas for future work.

## 1) Token Normalization

Core idea: normalize traced token-level signals per timestep across rollouts in the same group, instead of only normalizing sequence outcomes once.

Why it helps: this can stabilize per-position credit assignment, especially when group members have different completion lengths and sparse outcome rewards.

Expected effect: less dominance from outlier rollouts at a single timestep, more balanced token-level learning signal.

## 2) Additive Normalization (std + tau)

Core idea: use an additive denominator for token normalization,

```text
normalized = (x - mu) / (sigma + tau)
```

instead of pure z-score style scaling.

Why it helps: prevents very small standard deviations from causing overly large normalized values.

Expected effect: smoother optimization and reduced sensitivity when group variance is tiny.

## 3) Flat Incorrect Trace

Core idea: keep decayed lambda trace for correct samples, but set incorrect samples to a flat trace (no positional decay).

Why it helps: avoids over-emphasizing late-token credit on incorrect trajectories where errors may be distributed across the full response.

Expected effect: broader corrective pressure across incorrect tokens, while preserving positional refinement on correct traces.

## 4) Lambda + Gamma Discount

Core idea: combine sequence-level gamma discounting with lambda token tracing.

- Sequence side: scale sequence outcome by gamma^K (length-sensitive penalty)
- Token side: redistribute that outcome with lambda trace over token positions

Why it helps: gamma controls the total reward magnitude by length, while lambda controls where credit lands over positions.

Expected effect: explicit length pressure plus structured temporal credit assignment.

## 5) Second Trace Lambda

Core idea: after token normalization, apply a second positional trace weighting (with its own strength parameter).

Why it helps: adds an extra shaping stage to tune how strongly early vs late tokens are emphasized after normalization.

Expected effect: finer control over temporal credit geometry than a single trace pass.

## 6) Gamma + Additive Tau

Core idea: combine discounted reasoning rewards (gamma close to 1, e.g. 0.999) with additive GRPO normalization,

```text
normalized = (x - mu) / (sigma + tau)
```

instead of dividing by sigma alone.

Why it helps: discounted reasoning can make within-group advantages close together, which can produce tiny sigma and unstable scaling. Adding tau keeps normalization well-behaved in low-variance groups.

Expected effect: less noisy advantage magnitudes, fewer update spikes, and smoother training when reward differences are subtle.

Script linkage: `scripts/train_gamma_additive_tau_test.job`.

## 7) GRPO-Lambda Group-Shortest Trace Base

Core idea: use a group-shared shortest-anchored trace base for lambda-style token credit, where the effective trace base is computed from the shortest response in the group plus alpha.

Why it helps: standard lambda tracing can implicitly favor long responses because they have more positions to accumulate shaped credit. Anchoring to the shortest group member makes the trace geometry more comparable across rollouts.

Expected effect: more length-robust credit assignment, less long-response bias inside each GRPO group, and cleaner comparisons between candidates of different lengths.

Script linkage: `scripts/train_grpo_lambda_group_shortest_trace_test.job`.

## 8) Discounted Reasoning Mixed-Groups-Only (GRPO)

Core idea: keep discounted reasoning under the same GRPO setting, but gate it so gamma discount applies only when a group is mixed (contains at least one correct and one incorrect sample).

Gating rule:

```text
R_i' = R_i * gamma^K_i, only if (correct_i == 1) and (group_is_mixed == 1)
R_i' = R_i, otherwise
```

Why it helps: all-correct groups already have low within-group learning signal for outcome ranking. Skipping extra gamma pressure there focuses updates on unresolved groups where relative discrimination is still needed.

Expected effect: reduced over-regularization on solved groups, while retaining length pressure in mixed groups where policy improvement is still active.

Script linkage: `scripts/train_gamma_mixed_groups_only_test.job`.

