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

### Exact Variant You Requested: Group-Shared Shortest-Anchored Trace

All samples in a group share the same exponential base, calibrated from the shortest rollout.
Longer rollouts simply continue that same trace for extra early tokens.

```text
lambda_eff = clip(alpha^(1 / max(T_min - 1, 1)) / gamma, lambda_min, lambda_max)
base       = gamma * lambda_eff
w_i,t      = base^(T_i - 1 - t)
A_i,t      = r_hat_i * w_i,t
```

Key property:
- For any two samples i and j, when aligned by distance-to-end d, they have identical weights: w_i(d) = w_j(d).
- Only longer samples have additional prefix tokens with d > T_min - 1.

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
