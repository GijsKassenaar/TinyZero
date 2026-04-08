# GRPO-Lambda Ideas Summary

Date: 2026-04-01

This note briefly summarizes the current GRPO-lambda ideas in TinyZero and proposes one new idea for future work.

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

## New Idea Proposal: Adaptive Lambda (Not Implemented)

Name: Adaptive Lambda

Goal: replace a fixed lambda with a group-aware smoothed lambda that adapts to rollout lengths, instead of always using the same exponential decay profile.

### Motivation

With fixed lambda, long rollouts can get very small early-token credit, while short rollouts can remain relatively dense. This can create inconsistent training pressure across groups with diverse response lengths.

### Variant A: Group-Length Smoothed Lambda

Use group length statistics to set an effective lambda:

```text
lambda_eff = f(mean(T_group), std(T_group))
```

where f is a bounded smoothing function and lambda_eff is clipped to [lambda_min, lambda_max].

Intuition: groups with very long responses can use a slightly higher lambda (slower decay) to avoid vanishing early-token credit.

### Variant B: Shortest-Rollout Anchored Lambda

Anchor lambda to the shortest valid rollout in the group:

```text
lambda_eff = g(T_min_group)
```

so the trace profile is calibrated against the shortest trajectory, reducing mismatch between short and long traces in the same group.

Intuition: use the shortest rollout as a stable reference so long-rollout traces do not become disproportionately sharp.

### Practical Constraints

- Keep lambda_eff clipped to a safe range.
- Update at group level (not per token) for stability.
- Log diagnostics: group lengths, lambda_eff, and resulting trace mass distribution.

Status: idea only; not implemented yet.

## Adaptive Lambda Addendum (Readable Worked Variants)

Date: 2026-04-08

This addendum expands the adaptive-lambda idea into concrete shortest-anchored trace functions.

### Shared Notation

Use the following notation:

```text
T_i      = rollout length of sample i
T_min    = min_i T_i
u_i,t    = t / (T_i - 1), if T_i > 1, else 1
r_hat_i  = sequence-level normalized outcome
A_i,t    = r_hat_i * w_i,t
```

Anchor requirement:

```text
For the shortest rollout i*:
w_i*,0        = alpha
w_i*,T_min-1  = 1
```

### Option 1: Shortest-Anchored Exponential (closest to current GRPO-lambda)

```text
lambda_eff = clip(alpha^(1 / max(T_min - 1, 1)) / gamma, lambda_min, lambda_max)
w_i,t      = (gamma * lambda_eff)^(T_i - 1 - t)
```

Notes:
- Minimal code-change option.
- Preserves the current exponential trace geometry.

### Option 2: Shortest-Anchored Linear Ramp

```text
a_i   = clip(alpha * (T_min / T_i)^p, a_min, 1)
w_i,t = a_i + (1 - a_i) * u_i,t
```

Notes:
- Shortest rollout gets exact linear interpolation from alpha to 1.
- Higher p makes longer rollouts harsher near the first tokens.

### Option 3: Shortest-Anchored Power Ramp (smooth generalization)

```text
a_i    = clip(alpha * (T_min / T_i)^p, a_min, 1)
eta_i  = 1 + beta * (1 - T_min / T_i)
w_i,t  = a_i + (1 - a_i) * (u_i,t ^ eta_i)
```

Notes:
- For the shortest rollout, eta_i = 1, so this reduces to linear.
- For longer rollouts, eta_i > 1 and credit shifts later smoothly.

### Option 4: Shortest-Anchored Logistic S-Curve

```text
a_i      = clip(alpha * (T_min / T_i)^p, a_min, 1)
kappa_i  = kappa_0 + kappa_1 * (1 - T_min / T_i)
s(u,k)   = (sigmoid(k * (u - 0.5)) - sigmoid(-k/2)) / (sigmoid(k/2) - sigmoid(-k/2))
w_i,t    = a_i + (1 - a_i) * s(u_i,t, kappa_i)
```

Notes:
- Enforces start/end constraints while staying very smooth.
- Good when linear/power traces feel too rigid.

### Optional Stabilizer: Trace-Mass Normalization

```text
w_norm_i,t = w_i,t * T_i / (sum_{k=0..T_i-1}(w_i,k) + eps)
```

Use w_norm_i,t in place of w_i,t if you want stable average trace mass across different shapes.

### Practical Starter Settings

- Option 1 (exponential): alpha in {0.2, 0.3}, lambda_min = 0.985, lambda_max = 0.9997.
- Option 2 (linear): alpha in {0.2, 0.3}, p in {1, 2}, a_min = 0.05.
- Option 3 (power): alpha = 0.25, p = 1, beta in {2, 4}, a_min = 0.05.
- Option 4 (logistic): alpha = 0.25, p = 1, kappa_0 in [2, 4], kappa_1 in [2, 8].

### Suggested First Ablation Slice

1. Option 1 (minimal code delta) vs current fixed-lambda baseline.
2. Option 2 (linear) for the most interpretable smooth trace.
3. Option 3 (power) to test stronger late-token shaping without exponential collapse.

Log per group:
- T_min, mean/std lengths, chosen shape parameters.
- Trace first-token, middle-token, and last-token stats.
