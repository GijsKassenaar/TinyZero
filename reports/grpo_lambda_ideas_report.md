# GRPO-Lambda Ideas Summary

Date: 2026-04-01

This note briefly summarizes the current GRPO-lambda ideas in TinyZero and proposes one new idea for future work.

## 1) Token Normalization

Core idea: normalize traced token-level signals per timestep across rollouts in the same group, instead of only normalizing sequence outcomes once.

Why it helps: this can stabilize per-position credit assignment, especially when group members have different completion lengths and sparse outcome rewards.

Expected effect: less dominance from outlier rollouts at a single timestep, more balanced token-level learning signal.

## 2) Additive Normalization (std + tau)

Core idea: use an additive denominator for token normalization,

\[
\text{normalized} = \frac{x - \mu}{\sigma + \tau}
\]

instead of pure z-score style scaling.

Why it helps: prevents very small standard deviations from causing overly large normalized values.

Expected effect: smoother optimization and reduced sensitivity when group variance is tiny.

## 3) Flat Incorrect Trace

Core idea: keep decayed lambda trace for correct samples, but set incorrect samples to a flat trace (no positional decay).

Why it helps: avoids over-emphasizing late-token credit on incorrect trajectories where errors may be distributed across the full response.

Expected effect: broader corrective pressure across incorrect tokens, while preserving positional refinement on correct traces.

## 4) Lambda + Gamma Discount

Core idea: combine sequence-level gamma discounting with lambda token tracing.

- Sequence side: scale sequence outcome by \(\gamma^K\) (length-sensitive penalty)
- Token side: redistribute that outcome with lambda trace over token positions

Why it helps: gamma controls the total reward magnitude by length, while lambda controls where credit lands over positions.

Expected effect: explicit length pressure plus structured temporal credit assignment.

## 5) Second Trace Lambda

Core idea: after token normalization, apply a second positional trace weighting (with its own strength parameter).

Why it helps: adds an extra shaping stage to tune how strongly early vs late tokens are emphasized after normalization.

Expected effect: finer control over temporal credit geometry than a single trace pass.

## New Idea Proposal: Adaptive Lambda (Not Implemented)

Name: Adaptive Lambda

Goal: replace a fixed \(\lambda\) with a group-aware smoothed \(\lambda\) that adapts to rollout lengths, instead of always using the same exponential decay profile.

### Motivation

With fixed \(\lambda\), long rollouts can get very small early-token credit, while short rollouts can remain relatively dense. This can create inconsistent training pressure across groups with diverse response lengths.

### Variant A: Group-Length Smoothed Lambda

Use group length statistics to set an effective lambda:

\[
\lambda_{\text{eff}} = f(\bar{T}_{\text{group}},\; \text{std}(T_{\text{group}}))
\]

where \(f\) is a bounded smoothing function and \(\lambda_{\text{eff}}\in[\lambda_{\min},\lambda_{\max}]\).

Intuition: groups with very long responses can use a slightly higher \(\lambda\) (slower decay) to avoid vanishing early-token credit.

### Variant B: Shortest-Rollout Anchored Lambda

Anchor lambda to the shortest valid rollout in the group:

\[
\lambda_{\text{eff}} = g(T_{\min,\text{group}})
\]

so the trace profile is calibrated against the shortest trajectory, reducing mismatch between short and long traces in the same group.

Intuition: use the shortest rollout as a stable reference so long-rollout traces do not become disproportionately sharp.

### Practical Constraints

- Keep \(\lambda_{\text{eff}}\) clipped to a safe range.
- Update at group level (not per token) for stability.
- Log diagnostics: group lengths, \(\lambda_{\text{eff}}\), and resulting trace mass distribution.

Status: idea only; not implemented yet.
