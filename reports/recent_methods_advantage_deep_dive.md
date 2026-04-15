# TinyZero Recent Methods Deep Dive

Date: 2026-04-14

This document is a second report focused on a smaller set of recent methods.
Each section starts from the corresponding section in the comprehensive report and expands with:
- explicit formulas,
- one compact numeric example showing advantage impact,
- a placeholder block for experiment outcomes.

## Todo List (Thoroughness Checklist)

- [x] Select the nine requested methods from the comprehensive report
- [x] Expand each method with clear equations
- [x] Add one quick numeric advantage example per method
- [x] Attach script mapping for each method
- [x] Add result placeholders under each section
- [x] Add loss agg mode section (token-mean vs seq-mean-token-mean)
- [ ] Fill all sections with measured metrics from runs
- [ ] Add cross-method comparison table after results are filled

## Shared Notation

For rollout i in group g and token t:

$$
R_i = \sum_t r_{i,t} m_{i,t}
$$

where $m_{i,t}\in\{0,1\}$ is the valid-token mask.

Group normalization:

$$
\mu_g = \frac{1}{|g|}\sum_{i\in g} R_i, \quad
\sigma_g = \sqrt{\frac{1}{|g|}\sum_{i\in g}(R_i-\mu_g)^2}
$$

$$
\hat{R}_i = \frac{R_i-\mu_g}{\sigma_g + \delta}
$$

Base lambda trace (end-aligned):

$$
d_{i,t} = (\gamma\lambda)^{T_i-1-t}
$$

Base GRPO-lambda token advantage:

$$
A_{i,t} = \hat{R}_i\, d_{i,t}\, m_{i,t}
$$

## Loss Aggregation Mode: token-mean vs seq-mean-token-mean

This section explains how per-token losses are reduced to one scalar in training.

Where they appear in recent scripts:
- token-mean is used in scripts/train_dapo_test.job
- seq-mean-token-mean is used in scripts/train_gamma_test.job and scripts/train_gamma_additive_tau_test.job

Let $\ell_{i,t}$ be token-level loss for sample $i$, token $t$, and $m_{i,t}\in\{0,1\}$ be the valid-token mask.
Define valid length $s_i = \sum_t m_{i,t}$.

token-mean:

$$
\mathcal{L}_{token} = \frac{\sum_i\sum_t \ell_{i,t} m_{i,t}}{\sum_i\sum_t m_{i,t}}
$$

seq-mean-token-mean:

$$
\mathcal{L}_{seqtok} = \frac{1}{N_{valid}}\sum_{i:s_i>0}\left(\frac{\sum_t \ell_{i,t} m_{i,t}}{s_i + \epsilon}\right)
$$

where $N_{valid}$ is number of non-empty sequences in the batch.

In distributed training, the implementation in core_algos.agg_loss also applies a data-parallel scaling factor after normalization.

Quick example:
- Sequence A has 8 valid tokens with average per-token loss 1.0, so token-sum is 8.
- Sequence B has 2 valid tokens with average per-token loss 2.0, so token-sum is 4.
- token-mean gives $(8+4)/(8+2)=1.2$.
- seq-mean-token-mean gives $(1.0+2.0)/2=1.5$.

Impact intuition:
- token-mean gives longer responses more influence on the batch loss.
- seq-mean-token-mean gives each response a more equal contribution regardless of length.

## 1) Lambda Incorrect Flat Trace

Source in comprehensive report: GRPO-lambda Variant A - Flat Incorrect Trace.

Primary script:
- scripts/train_grpo_lambda_flat_incorrect_trace_test.job

Core modification:

$$
A_{i,t} = \hat{R}_i \cdot \big(c_i d_{i,t} + (1-c_i)\cdot 1\big) \cdot m_{i,t}
$$

where $c_i=1$ for correct and $c_i=0$ for incorrect.

Quick example:
- Assume $\hat{R}_{correct}=+1$, $\hat{R}_{incorrect}=-1$, $\gamma\lambda=0.99$, $T=4$.
- Decay trace for correct: $[0.970, 0.980, 0.990, 1.000]$.
- Incorrect flat trace: $[1,1,1,1]$.
- Advantages:
  - Correct: $[+0.970,+0.980,+0.990,+1.000]$
  - Incorrect: $[-1.000,-1.000,-1.000,-1.000]$

Impact intuition:
- Incorrect samples no longer get reduced early-token magnitude from trace decay.

Results placeholder:
- Run:
- Flat-incorrect vs baseline accuracy:
- Effect on incorrect trajectory correction:
- Notes:

## 2) Lambda + Gamma (Lambda Sequence Gamma Discount)

Source in comprehensive report: GRPO-lambda Variant B - Sequence Gamma Discount + Reasoning-only Trace.

Primary script:
- scripts/train_lambda_gamma.job

Core modification:

$$
R_i^{seq} =
\begin{cases}
R_i\,\gamma_s^{K_i}, & c_i=1 \\
R_i, & c_i=0
\end{cases}
$$

then group normalization and lambda trace proceed as usual:

$$
\hat{R}_i = \frac{R_i^{seq}-\mu_g}{\sigma_g+\delta},\quad
A_{i,t}=\hat{R}_i d_{i,t}m_{i,t}
$$

Quick example:
- With script-like $\gamma_s=0.99999999$:
  - $K=200$: $\gamma_s^K \approx 0.999998$
  - $K=2000$: $\gamma_s^K \approx 0.999980$
- The discount is intentionally very mild at this gamma.

Impact intuition:
- Adds length sensitivity at sequence level, but strength depends heavily on gamma choice.

Results placeholder:
- Run:
- Discounted sequence reward stats:
- Sensitivity to reasoning length:
- Notes:

## 3) Lambda 0.999 Token Normalization

Source in comprehensive report: GRPO-lambda Variant C - Token Normalization.

Primary script:
- scripts/train_grpo_lambda_token_normalization_test.job

Core modification:

First compute traced signal:

$$
Z_{i,t} = R_i\, d_{i,t}
$$

Then normalize per timestep across group rollouts:

$$
\mu_t = \frac{1}{|g|}\sum_{i\in g} Z_{i,t}, \quad
\sigma_t = \sqrt{\frac{1}{|g|}\sum_{i\in g}(Z_{i,t}-\mu_t)^2}
$$

$$
A_{i,t} = \frac{Z_{i,t}-\mu_t}{\sigma_t+\delta}\,m_{i,t}
$$

Quick example:
- At one timestep, let $Z=[0.9, 0.6, 0.3]$.
- $\mu_t=0.6$, $\sigma_t\approx0.245$.
- Normalized advantages: $[+1.225, 0, -1.225]$.

Impact intuition:
- Standardizes token credit per position, reducing domination by outlier rollouts at a single token index.

Results placeholder:
- Run:
- Token-wise variance before and after normalization:
- Accuracy and stability:
- Notes:

## 4) GRPO Lambda Second Trace

Source in comprehensive report: GRPO-lambda Variant E - Second Trace After Token Norm.

Primary script:
- scripts/train_grpo_lambda_second_trace_test.job

Core modification:

After token normalization $A^{norm}_{i,t}$:

$$
A_{i,t} = A^{norm}_{i,t}\, d_{i,t}^{\alpha}\, m_{i,t}
$$

with $\alpha=10$ in the script.

Quick example:
- Suppose $A^{norm}_{i,t}=[1,1,1,1]$.
- With $d=[0.970,0.980,0.990,1.000]$ and $\alpha=10$:
  - $d^{10}\approx[0.740,0.818,0.904,1.000]$
- Final advantages become $[0.740,0.818,0.904,1.000]$.

Impact intuition:
- Strong post-normalization temporal shaping, especially suppressing early tokens when alpha is high.

Results placeholder:
- Run:
- Position-wise advantage redistribution:
- Accuracy and response-length behavior:
- Notes:

## 5) Lambda 0.99 Reasoning Only

Source relation: reasoning-only trace component used in lambda variant controls.

Closest script surface:
- scripts/train_lambda_gamma.job (reasoning_only_discount_trace_enable=True)

Core modification:

Define reasoning mask $q_{i,t}\in\{0,1\}$.
Only reasoning tokens receive trace decay:

$$
d^{reason}_{i,t} = q_{i,t}d_{i,t} + (1-q_{i,t})\cdot 1
$$

$$
A_{i,t}=\hat{R}_i d^{reason}_{i,t} m_{i,t}
$$

Quick example:
- Let $\lambda=0.99$, $\gamma=1$, $d=[0.961,0.970,0.980,0.990,1.000]$.
- Reasoning mask $q=[1,1,1,0,0]$.
- Effective trace becomes $[0.961,0.970,0.980,1.000,1.000]$.

Impact intuition:
- Decay pressure is concentrated on reasoning span, while answer tokens are preserved from extra decay.

Results placeholder:
- Run:
- Reasoning-span advantage magnitude change:
- Final answer token behavior:
- Notes:

## 6) Lambda GRPO Token Norm Additive Tau

Source in comprehensive report: GRPO-lambda Variant D - Token-Norm Additive Tau.

Primary script:
- scripts/train_grpo_lambda_token_normalization_additive_tau_test.job

Core modification:

Token-wise normalization denominator changes from $\sigma_t+\delta$ to $\sigma_t+\tau$:

$$
A_{i,t}=\frac{Z_{i,t}-\mu_t}{\sigma_t+\tau}m_{i,t}
$$

Quick example:
- One timestep: $Z=[0.91,0.90,0.89]$.
- $\mu_t=0.90$, $\sigma_t\approx0.0082$.
- Centered value for top rollout is $0.01$.
- Without tau: $0.01/0.0082\approx1.22$.
- With $\tau=0.02$: $0.01/(0.0082+0.02)\approx0.35$.

Impact intuition:
- Prevents oversized normalized advantages in low-variance timesteps.

Results placeholder:
- Run:
- Low-variance timestep stability:
- Clip fraction and loss-spike behavior:
- Notes:

## 7) GRPO Lambda Group Shortest Alpha Trace

Source in comprehensive report: GRPO-lambda Variant F - Group-Shortest Trace Base.

Primary script:
- scripts/train_grpo_lambda_group_shortest_trace_test.job

Core modification:

For group g, let shortest valid length be $L_{min,g}$ and configured alpha be $\alpha_s$.
Shared base:

$$
b_g = \alpha_s^{1/(L_{min,g}-1)}
$$

Trace:

$$
d_{i,t}=b_g^{e_{i,t}}, \quad e_{i,t}=T_i-1-t
$$

Advantage:

$$
A_{i,t}=\hat{R}_i d_{i,t} m_{i,t}
$$

Quick example:
- $L_{min}=100$, $\alpha_s=0.15$.
- $b=0.15^{1/99}\approx0.981$.
- Shortest sample first-token weight: $0.981^{99}=0.15$.
- Longer sample with $T=200$ first-token weight: $0.981^{199}\approx0.022$.

Impact intuition:
- Creates a common group-level decay geometry anchored to shortest completion.

Results placeholder:
- Run:
- Length-bias diagnostics:
- Group-wise calibration behavior:
- Notes:

## 8) Gamma 0.999 Additive Tau

Source in comprehensive report: Discounted Reasoning + Plain GRPO Additive Normalization.

Primary script:
- scripts/train_gamma_additive_tau_test.job

Core modification (plain GRPO path):

Correct samples are discounted by reasoning length:

$$
R_i' =
\begin{cases}
R_i\gamma^{K_i}, & c_i=1 \\
R_i, & c_i=0
\end{cases}
$$

Then normalized with additive tau:

$$
\hat{R}_i = \frac{R_i'-\mu_g}{\sigma_g+\tau}, \quad
A_{i,t}=\hat{R}_i m_{i,t}
$$

Quick example:
- Discounted sequence rewards in one group: $[0.74,0.73,0.72,0.71]$.
- $\mu_g=0.725$, $\sigma_g\approx0.0112$.
- Top centered value: $0.015$.
- Without tau: $0.015/0.0112\approx1.34$.
- With $\tau=0.02$: $0.015/(0.0112+0.02)\approx0.48$.

Impact intuition:
- Gamma introduces length pressure; additive tau limits amplification when spread is tiny.

Results placeholder:
- Run:
- Mean reasoning discount factor:
- Advantage magnitude distribution:
- Notes:

## 9) Gamma 0.99 Additive Tau + Incorrect Penalty

Requested target method:
- gamma 0.99 additive tau plus incorrect penalty

Closest existing surface:
- scripts/train_gamma_additive_tau_test.job (same structure, currently gamma=0.999 and incorrect penalty enabled)

Core modification:

Incorrect penalty first:

$$
R_i^{pen} = c_iR_i + (1-c_i)p, \quad p=-1
$$

Then discount only correct trajectories:

$$
R_i' =
\begin{cases}
R_i^{pen}\gamma^{K_i}, & c_i=1 \\
R_i^{pen}, & c_i=0
\end{cases}
$$

Then additive-tau normalization:

$$
\hat{R}_i=\frac{R_i'-\mu_g}{\sigma_g+\tau}, \quad A_{i,t}=\hat{R}_i m_{i,t}
$$

Quick example (gamma=0.99):
- Two correct samples with raw score 1 and $K=[30,120]$: discounted to $[0.740,0.299]$.
- Two incorrect samples penalized to $[-1,-1]$.
- Group scores: $[0.740,0.299,-1,-1]$.
- $\mu_g\approx-0.240$, $\sigma_g\approx0.776$.
- With $\tau=0.02$, normalized scores become approximately:
  - $[+1.232, +0.678, -0.955, -0.955]$.

Impact intuition:
- Combines strong incorrect suppression with stronger length pressure than gamma=0.999.

Results placeholder:
- Run:
- Accuracy and failure-mode breakdown:
- Sensitivity to gamma and tau:
- Notes:

## Next Fill-in Table

- Add run IDs and checkpoint paths for each section.
- Add one metric trio per section: accuracy, response length, and throughput or token cost.
- Add one stability trio per section: clip fraction, grad norm, and reward variance.
- Add one sentence conclusion per method after metrics are filled.
