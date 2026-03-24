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

In this report's LEAD calculations, $\mu_{len}$ and $\sigma_{len}$ are computed using only correct responses in the group.

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
r_i = \gamma^{K_i},\quad \gamma=0.999
$$

Numerically:

- rewards $\approx [0.53778, 0.50645, 0.47694, 0.45366]$
- normalized group advantages:
    - $[1.2086, 0.3493, -0.4599, -1.0981]$

Interpretation: shorter correct traces receive larger positive sequence advantage.

### D) GRPO-lambda token tracing from vanilla GRPO only (short version)

Start from vanilla GRPO sequence advantages (section A): all are zero.

$$
A_i^{seq}=0 \quad\Rightarrow\quad A_{i,t}^{\lambda}=A_i^{seq}\cdot 0.99^{T_i-1-t}=0
$$

So for this all-correct/equal-reward group, lambda cannot create signal; it only redistributes existing signal.

---

## Worked Example 2: Same lengths, but 2 correct and 2 incorrect

Setup:

- Lengths $K=[620,680,740,790]$
- Correctness $c=[1,1,0,0]$
- $\lambda=0.99$, $\gamma=0.999$

### A) LEAD (no difficulty reweighting, correct-only length z-score)

Using $\alpha=0.1$, incorrect penalty $=-1.0$:

- LEAD rewards: $[1.1052,\ 0.9048,\ -1.0,\ -1.0]$
- Normalized advantages: $[0.9502,\ 0.7776,\ -0.8639,\ -0.8639]$

### B) Gamma-discounted reasoning

Using $r_i=\gamma^{K_i}\cdot c_i$:

- Rewards: $[0.53778,\ 0.50645,\ 0,\ 0]$
- Normalized advantages: $[0.9172,\ 0.8133,\ -0.8652,\ -0.8652]$

### C) GRPO-lambda (vanilla GRPO + lambda)

Vanilla GRPO sequence rewards are binary: $[1,1,0,0]$.

- Vanilla normalized sequence advantages: $[0.8660,\ 0.8660,\ -0.8660,\ -0.8660]$

Apply token trace

$$
A_{i,t}^{\lambda}=A_i^{seq}\cdot 0.99^{T_i-1-t}
$$

First/mid/last token values:

- Sample 1 ($T=620$): $[0.001721,\ 0.038410,\ 0.8660]$
- Sample 2 ($T=680$): $[0.000942,\ 0.028412,\ 0.8660]$
- Sample 3 ($T=740$): $[-0.000515,\ -0.021016,\ -0.8660]$
- Sample 4 ($T=790$): $[-0.000312,\ -0.016347,\ -0.8660]$

---

## Worked Example 3: Longer rollouts (~2000) with higher length variance, 2 correct and 2 incorrect

Setup:

- Lengths $K=[1600,1900,2200,2500]$
- Correctness $c=[1,0,1,0]$
- $\lambda=0.99$, $\gamma=0.999$

### A) LEAD (no difficulty reweighting, correct-only length z-score)

Using $\alpha=0.1$, incorrect penalty $=-1.0$:

- LEAD rewards: $[1.1052,\ -1.0,\ 0.9048,\ -1.0]$
- Normalized advantages: $[0.9502,\ -0.8639,\ 0.7776,\ -0.8639]$

### B) Gamma-discounted reasoning

Using $r_i=\gamma^{K_i}\cdot c_i$:

- Rewards: $[0.20174,\ 0,\ 0.11068,\ 0]$
- Normalized advantages: $[1.2674,\ -0.8007,\ 0.3340,\ -0.8007]$

### C) GRPO-lambda (vanilla GRPO + lambda)

Vanilla GRPO sequence rewards are binary: $[1,0,1,0]$.

- Vanilla normalized sequence advantages: $[0.8660,\ -0.8660,\ 0.8660,\ -0.8660]$

Apply token trace

$$
A_{i,t}^{\lambda}=A_i^{seq}\cdot 0.99^{T_i-1-t}
$$

First/mid/last token values:

- Sample 1 ($T=1600$): $[9.08\times10^{-8},\ 2.79\times10^{-4},\ 0.8660]$
- Sample 2 ($T=1900$): $[-4.45\times10^{-9},\ -6.18\times10^{-5},\ -0.8660]$
- Sample 3 ($T=2200$): $[2.18\times10^{-10},\ 1.37\times10^{-5},\ 0.8660]$
- Sample 4 ($T=2500$): $[-1.07\times10^{-11},\ -3.03\times10^{-6},\ -0.8660]$

Interpretation: with long traces and $\lambda<1$, early-token lambda credit becomes tiny. Signal is concentrated near later tokens.

---

## Practical takeaway in this codebase

- LEAD and discounted reasoning mostly alter sequence-level reward shaping before GRPO normalization.
- GRPO-lambda alters token-level credit assignment after group normalization via backward traces.
- Combining discounted reasoning with GRPO-lambda is coherent: discounting changes per-sequence outcome magnitude, then lambda redistributes that signal over token positions.

---

## Qualitative comparison of the three algorithms

### GRPO-lambda: implicit discounting of credit, not reward

- Nature of discount: implicit.
- What is discounted: token-level credit assignment over position.
- What is not discounted: sequence reward itself.
- Effect: later tokens receive stronger credit, earlier tokens weaker credit, but total reward signal is not directly shrunk.
- Practical intuition: this behaves like a mild temporal discount on gradients, not a hard penalty on long responses.

### GRPO-LEAD: relative, group-based length discounting

- Nature of discount: relative.
- What is discounted: score for correct samples based on z-scored length within the group.
- Reference frame: other rollouts for the same prompt.
- Effect: length penalties are assigned in perspective to group behavior, so "too long" is defined relative to peer completions.
- Practical intuition: this is a context-aware length pressure, not a fixed global token tax.

### Gamma-discounted reasoning: naive global discounting

- Nature of discount: direct/global.
- What is discounted: reward magnitude itself through $\gamma^K$.
- Reference frame: absolute reasoning-token count, independent of group-relative statistics.
- Effect: every additional reasoning token reduces reward multiplicatively.
- Practical intuition: this is a straightforward token-cost objective that can over-penalize long reasoning if $\gamma$ is too small.

### Side-by-side takeaway

- GRPO-lambda changes where credit goes.
- LEAD changes how samples are ranked relative to peers.
- Gamma discount changes the reward scale directly as length grows.

In short: lambda is the most implicit, LEAD is relative, gamma is the most explicit and naive length penalizer.

---

## References

- [grpo-lambda] Prasanna Parthasarathi, Mathieu Reymond, Boxing Chen, Yufei Cui, and Sarath Chandar. *GRPO-$\lambda$: Credit Assignment improves LLM Reasoning* (2025). arXiv:2510.00194. https://arxiv.org/abs/2510.00194
- [grpo-LEAD] Jixiao Zhang and Chunsheng Zuo. *GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models* (2025). arXiv:2504.09696. https://arxiv.org/abs/2504.09696
- [naive-discount] Alex Ayoub, Kavosh Asadi, Dale Schuurmans, Csaba Szepesvari, and Karim Bouyarmane. *Learning to Reason Efficiently with Discounted Reinforcement Learning* (2025). arXiv:2510.23486. https://arxiv.org/abs/2510.23486
