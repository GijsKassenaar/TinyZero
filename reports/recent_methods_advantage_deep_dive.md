# Recent Methods


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

- Lambda also discount the neagitve advantage of incorrect traces. with a flat trace for incorrect answers the full rollout is counted inccorect

Results:


- Accuracy: 0.6
- Effect on length: Answers become longer especially easy ones
- Notes: Since neagtive advantages are not discounted there is an imbalance in the loss which causes instability. perfromance is much worse.

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

- Combines the effect of gamma and lambda discount. The learning effect of lambda with the more effective lenght penalties of gamma.

Results:


- Accuracy: 0.67
- Effect on length: does not lower length compared to lambda only.
- Notes: Used a very small gamma discount for minimal effect. did not lower length but did lower accuracy.

## 3) Lambda 0.99 Reasoning Only

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

Results:


- Accuracy: 0.58
- Effect on length: Heavily discount reasoning but answer become extremly long
- Notes: model is incentivized to stop reasoning tokens and instead reason within the answer.



## 4) GRPO Lambda Group Shortest Alpha Trace

Source in comprehensive report: GRPO-lambda Variant F - Group-Shortest Trace Base.

Primary script:

- scripts/train_grpo_lambda_group_shortest_trace_test.job

Core modification:

For group g, let shortest valid length be $L_{min,g}$ and configured alpha be $\alpha_s$.
Adaptive lambda-token base:

$$
\lambda_{token,g} = \alpha_s^{1/(L_{min,g}-1)}
$$

Trace:

$$
d_{i,t}=\lambda_{token,g}^{e_{i,t}}, \quad e_{i,t}=T_i-1-t
$$

Advantage:

$$
A_{i,t}=\hat{R}_i d_{i,t} m_{i,t}
$$

Quick example:

- $L_{min}=100$, $\alpha_s=0.15$.
- $\lambda_{token,g}=0.15^{1/99}\approx0.981$.
- Shortest sample first-token weight: $0.981^{99}=0.15$.
- Longer sample with $T=200$ first-token weight: $0.981^{199}\approx0.022$.

Impact intuition:

- Instead of having the same lambda for each group, adapt it on the shortest answer in group, so that for the shortest correct answers in the group there is some signal for every token. for long answers this means less discounting, for shorter answers more discounting

Results:


- Accuracy: 0.68
- Effect on length: lenght becomes higher compared to lambda
- Notes: expected rise in accuracy since for long answers on difficult questions the discounting is less strong. actual perfromace is slightly worse.

## 5) Gamma 0.999 Additive Tau



Core modification (plain GRPO):

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

- Gamma introduces length pressure; tau basically dampens the advantages if the reward differnce is small.

Results:


- Accuracy: 0.66 ,for tau=0.02
- Effect on length: longer than gamma but still much lower then gamma and baseline
- Notes: Works great. increases gamma perfromance. still need to fin optimal tau

## 6) Gamma 0.99 Additive Tau + Incorrect Penalty

Requested target method:

- gamma 0.99 additive tau plus incorrect penalty

Closest existing surface:

- scripts/train_gamma_additive_tau_test.job (same structure, currently gamma=0.999 and incorrect penalty enabled)

Core modification:



Incorrect answers are given reward of -1 instead of 0:

$$
R_i' =
\begin{cases}
R_i^{pen}\gamma^{K_i}, & c_i=1 \\
-1, & c_i=0
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

- incorrect penalty comes from the LEAD method. gamma discounts can set reward for long rollouts to close to  0, this creates a bigger gap between long answers and incorrect answers

Results:


- Accuracy: 0.65
- Effect on length: no meaningful differnece compared to no incorrect penalty
- Notes: surprsingly no positve effect for longer answers. slighty decreases performance across the board


