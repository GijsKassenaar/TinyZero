# Report: Discount Factors and Length Penalties in GRPO Variants

## 1) GRPO-lambda (arXiv:2510.00194)

GRPO-lambda adds token-level temporal credit assignment to critic-free GRPO. The important difference versus vanilla GRPO is that the sequence-level group-normalized signal is not applied uniformly to all valid tokens. Instead, it is traced backward over token positions with a decay factor.

### How the trace works in practice

For a sample with valid response length $T$, token index $t \in \{0,\dots,T-1\}$, and normalized group outcome $\hat{r}_i$, GRPO-lambda in this report is written as a pure lambda trace:

$$
A_{i,t} = \hat{r}_i \cdot \lambda^{T-1-t}
$$

This means:

- Late tokens get large magnitude (exponent near 0).
- Early tokens get smaller magnitude (large exponent).
- The sign comes from $\hat{r}_i$; the trace only changes magnitude over position.

So it is still outcome-supervised, but no longer position-uniform. (Implementation note: your code path currently uses $\gamma\lambda$ as the decay base; for the calculation section below we set $\gamma=1$ to isolate lambda.)

### How this differs from vanilla GRPO

Vanilla GRPO (outcome form) effectively does:

$$
A_{i,t}^{GRPO} = \hat{r}_i \cdot \mathbf{1}_{t < T}
$$

All valid tokens get the same advantage scalar. In GRPO-lambda, this is replaced by a position-dependent scalar per token. That is the key behavioral difference.

### Core implementation snippets in this codebase

Snippet A: backward trace exponent and token advantages

```python
# verl/trainer/ppo/core_algos.py
valid_token_lengths = mask.sum(dim=-1, keepdim=True)
token_positions = (torch.cumsum(mask, dim=-1) - 1.0).clamp(min=0.0)
exponents = (valid_token_lengths - 1.0 - token_positions).clamp(min=0.0)

decay_base = rewards.new_tensor(gamma * lam)
decay_trace = torch.pow(decay_base, exponents)

token_advantages = normalized_reward.unsqueeze(-1) * decay_trace * mask
```

Snippet B: GRPO-lambda estimator selection in trainer

```python
# verl/trainer/ppo/ray_trainer.py
elif adv_estimator == AdvantageEstimator.GRPO_LAMBDA:
    advantages, returns = core_algos.compute_grpo_lambda_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        gamma=gamma,
        lam=lam,
    )
```

---

## 2) GRPO-LEAD (arXiv:2504.09696)

GRPO-LEAD changes group outcome shaping before normalization using three ideas:

1. Length-dependent reward for correct responses.
2. Explicit penalty for incorrect responses.
3. Difficulty-aware weighting of normalized advantages.

### Core equations

Length-standardized score within a group:

$$
z_i = \frac{|o_i|-\mu_{len}}{\sigma_{len}+\epsilon}
$$

Shaped reward:

$$
R_i =
\begin{cases}
\exp(-\alpha z_i), & \text{if correct}\\
\text{incorrect\_penalty}, & \text{if incorrect}
\end{cases}
$$

Difficulty weight from group correctness ratio $\rho_q$:

$$
w = \exp\left(-\frac{(\rho_q-\tau)^2}{2\beta^2}\right)
$$

Final normalized advantage is multiplied by this weight.

### Core implementation snippets in this codebase

Snippet A: LEAD score shaping in GRPO path

```python
# verl/trainer/ppo/core_algos.py
z = (group_lengths - mean_len) / (std_len + lead_epsilon)
group_rewards = torch.where(
    group_correct > 0.5,
    torch.exp(-lead_alpha * z),
    torch.full_like(z, incorrect_penalty),
)
```

Snippet B: difficulty-aware group weighting

```python
# verl/trainer/ppo/core_algos.py
acc_avg = group_correct.mean()
weight = torch.exp(-((acc_avg - lead_tau) ** 2) / (2.0 * beta_sq))
id2weight[group_id] = weight
...
scores[i] = scores[i] * id2weight[index[i]]
```

---

## 3) Discounted Reasoning GRPO (arXiv:2510.23486)

This method discounts reward by reasoning-token count to induce a small token-cost effect:

$$
R(\tau) = \gamma^{K(\tau)} r_e(\tau) + r_f(\tau)
$$

In your implementation, discounting is applied to reward using tokens inside think or thinking spans.

### Core implementation snippets in this codebase

Snippet A: extract reasoning length from think tags

```python
# verl/trainer/ppo/discounted_reasoning.py
for match in _THINKING_TAG_PATTERN.finditer(response_text):
    inner_text = match.group(1)
    if inner_text:
        reasoning_token_count += _tokenize_text_length(inner_text, tokenizer)
```

Snippet B: apply gamma^K to reward

```python
# verl/trainer/ppo/discounted_reasoning.py
discount_factors = torch.pow(
    torch.full_like(reasoning_lengths_tensor, gamma, dtype=torch.float32),
    reasoning_lengths_tensor,
)
discounted_reward_tensor = reward_tensor * discount_factors.unsqueeze(-1)
```

Snippet C: injection point in training loop

```python
# verl/trainer/ppo/ray_trainer.py
reward_tensor, discounted_reasoning_metrics = apply_reasoning_reward_discount(
    batch=batch,
    reward_tensor=reward_tensor,
    tokenizer=self.tokenizer,
    discount_cfg=discounted_reasoning_cfg,
)
```

---

## Worked Example: One GRPO group, size 4, all correct, lengths 600 to 800

Assume one prompt with four completions:

- Lengths $K = [620, 680, 740, 790]$
- All are correct
- Requested parameter for tracing: $\lambda=0.99$ (no $\gamma$ in the lambda trace calculation)

### A) Vanilla GRPO baseline

If all rewards are identical ($r_i = 1$), then z-score normalization within group yields:

$$
A_i = \frac{r_i - \mu}{\sigma + \epsilon} = 0
$$

All sequence advantages are zero, so token advantages are zero.

### B) LEAD shaping effect (no difficulty reweighting)

Using LEAD shaping with $\alpha=0.1$ only (skip difficulty reweighting to focus on length):

- $z \approx [-1.3718, -0.4311, 0.5095, 1.2934]$
- shaped rewards $\approx [1.1470, 1.0441, 0.9503, 0.8787]$
- normalized advantages:
  - $[1.2200, 0.3353, -0.4699, -1.0855]$

Interpretation: even with all-correct outcomes, LEAD creates ranking pressure toward concise correct traces.

### C) Discounted reasoning effect (gamma only)

Sequence rewards become:

$$
r_i = \gamma^{K_i},\quad \gamma=0.9999
$$

Numerically:

- rewards $\approx [0.93988, 0.93426, 0.92867, 0.92404]$
- normalized group advantages:
  - $[1.1899, 0.3709, -0.4431, -1.1177]$

Interpretation: shorter correct traces receive larger positive sequence advantage.

### D) GRPO-lambda token tracing from vanilla GRPO only (full calculation for all 4 examples)

Start from vanilla GRPO sequence advantages (section A). In this all-correct group with identical reward $r_i=1$:

$$
A_i^{seq} = \frac{r_i-\mu}{\sigma+\epsilon} = 0 \quad \forall i \in \{1,2,3,4\}
$$

Then apply lambda-only trace with $\lambda=0.99$:

$$
A_{i,t}^{\lambda} = A_i^{seq} \cdot 0.99^{T_i-1-t}
$$

Because each $A_i^{seq}=0$, every token advantage remains 0.

1. Example 1: $T_1=620$, $A_1^{seq}=0$
$$
\begin{aligned}
A_{1,\text{first}} &= 0\cdot 0.99^{619}=0,\\
A_{1,\text{mid}}   &= 0\cdot 0.99^{310}=0,\\
A_{1,\text{last}}  &= 0\cdot 0.99^{0}=0.
\end{aligned}
$$

2. Example 2: $T_2=680$, $A_2^{seq}=0$
$$
\begin{aligned}
A_{2,\text{first}} &= 0\cdot 0.99^{679}=0,\\
A_{2,\text{mid}}   &= 0\cdot 0.99^{340}=0,\\
A_{2,\text{last}}  &= 0\cdot 0.99^{0}=0.
\end{aligned}
$$

3. Example 3: $T_3=740$, $A_3^{seq}=0$
$$
\begin{aligned}
A_{3,\text{first}} &= 0\cdot 0.99^{739}=0,\\
A_{3,\text{mid}}   &= 0\cdot 0.99^{370}=0,\\
A_{3,\text{last}}  &= 0\cdot 0.99^{0}=0.
\end{aligned}
$$

4. Example 4: $T_4=790$, $A_4^{seq}=0$
$$
\begin{aligned}
A_{4,\text{first}} &= 0\cdot 0.99^{789}=0,\\
A_{4,\text{mid}}   &= 0\cdot 0.99^{395}=0,\\
A_{4,\text{last}}  &= 0\cdot 0.99^{0}=0.
\end{aligned}
$$

Interpretation: this isolates the exact role of lambda. Lambda only redistributes existing sequence advantage over token positions; it does not create non-zero signal when vanilla GRPO sequence advantages are all zero.

---

## Practical takeaway in this codebase

- LEAD and discounted reasoning mostly alter sequence-level reward shaping before GRPO normalization.
- GRPO-lambda alters token-level credit assignment after group normalization via backward traces.
- Combining discounted reasoning with GRPO-lambda is coherent: discounting changes per-sequence outcome magnitude, then lambda redistributes that signal over token positions.
