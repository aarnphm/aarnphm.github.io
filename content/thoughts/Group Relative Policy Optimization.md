---
date: '2025-04-26'
description: Estimate an advantage from several answers to the same question, then use it in a clipped policy update without a learned value critic.
id: Group Relative Policy Optimization
modified: 2026-09-15 09:16:38 GMT-04:00
tags:
  - ml
title: Group Relative Policy Optimization
---

GRPO estimates a baseline from a group of answers to the same question. This removes the learned value critic used in [[thoughts/Policy gradient|PPO]]. DeepSeekMath still uses a reward model to score answers and a reference policy to regularize the update [@shao2024deepseekmathpushinglimitsmathematical, §4.1].

## group-relative advantage

For outcome supervision, sample $G$ answers $o_i$ from the rollout policy $\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)$ fixed for this update and score them with rewards $r_i$. Each token in answer $i$ receives the same advantage:

$$
\bar r=\frac1G\sum_{j=1}^{G}r_j,
\qquad
\hat A_{i,t}=\frac{r_i-\bar r}{\operatorname{std}(r_1,\ldots,r_G)}.
$$

For example, rewards $(0,0,1,1)$ give advantages $(-1,-1,1,1)$ using the population standard deviation. The signal depends on which other answers were sampled. If all rewards are equal, the displayed normalization divides by zero. An implementation needs an explicit rule, such as adding a small positive denominator stabilizer. Since every numerator is zero, that group gets zero advantages. The KL term can still contribute to the update.

## objective

Let $h_{i,t}=(q,o_{i,<t})$ and define the token probability ratio

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid h_{i,t})}
     {\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid h_{i,t})}.
$$

Let $P(Q)$ be the question distribution, $|o_i|$ the output token count, and $\varepsilon>0$ the clipping parameter. Using exact conditional KL for the regularizer, the clipped group objective is:

$$
\begin{aligned}
\mathcal J_{\mathrm{GRPO}}(\theta)
&=\mathbb E_{\substack{q\sim P(Q)\\o_{1:G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}}
\left[\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}(\theta)\right],\\
\ell_{i,t}(\theta)
&=\min\!\left(
\rho_{i,t}\hat A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}
\right)-\beta D_{i,t},\\
D_{i,t}
&=D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid h_{i,t})\,\|\,
\pi_{\mathrm{ref}}(\cdot\mid h_{i,t})\right).
\end{aligned}
$$

Update $\pi_\theta$ to maximize this objective. Clipping caps this objective's reward incentive once the ratio moves far enough in the favored direction. The updated policy can still cross the clipping range. The coefficient $\beta$ weights the penalty for departing from the reference policy. The rollout and reference policies have separate roles. The equations use exact KL notation; DeepSeekMath's equation 4 supplies its sampled-token estimator.
