---
created: "2025-12-17"
date: "2025-12-17"
description: "Policy gradient RL methods: REINFORCE, baselines, TRPO, PPO, natural gradient"
id: Policy gradient
modified: 2025-12-17 00:59:20 GMT-05:00
published: "2017-03-01"
socials:
  wikipedia: https://en.wikipedia.org/wiki/Policy_gradient_method
tags:
  - ml
  - rl
  - training
title: Policy gradient
---

_Class of reinforcement learning algorithms._

**Policy gradient methods** are a class of **reinforcement learning** algorithms.

Policy gradient methods are a sub-class of policy optimization methods. Unlike value-based methods (which learn a value function to derive a policy), policy optimization methods directly learn a **policy function** $\pi$ that selects actions without consulting a value function. For policy gradient to apply, the policy function is parameterized as a differentiable $\pi_\theta$ with parameters $\theta$.

## Overview

In policy-based RL, the **actor** is a parameterized policy function $\pi_\theta$, where $\theta$ are the parameters of the actor. The actor takes the state $s$ and produces a probability distribution

$$
\pi_\theta(\cdot\mid s).
$$

- **Discrete actions:**

$$
\sum_a \pi_\theta(a\mid s)=1.
$$

- **Continuous actions:**

$$
\int_a \pi_\theta(a\mid s)\,\mathrm da=1.
$$

The goal of policy optimization is to find $\theta$ that maximizes the expected episodic reward

$$
J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t\in 0:T}\gamma^t R_t\;\Big|\;S_0=s_0\right],
$$

where:

- $\gamma$ is the discount factor,
- $R_t$ is the reward at step $t$,
- $s_0$ is the starting state,
- $T$ is the time horizon (possibly infinite).

The **policy gradient** is

$$
\nabla_\theta J(\theta).
$$

Different policy gradient methods stochastically estimate $\nabla_\theta J(\theta)$ in different ways. The goal is to iteratively maximize $J(\theta)$ by **gradient ascent**; since the key part is stochastic estimation, these are also studied under "Monte Carlo gradient estimation".

---

## REINFORCE

### Policy gradient

The **REINFORCE algorithm** (Williams, 1992) was the first policy gradient method. It is based on the identity

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t\in 0:T}\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\;\sum_{t\in 0:T}(\gamma^t R_t)\;\Big|\;S_0=s_0\right],
$$

which can be improved via the **causality trick**:

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t\in 0:T}\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\;\sum_{\tau\in t:T}(\gamma^\tau R_\tau)\;\Big|\;S_0=s_0\right].
$$

#### Lemma

**Lemma.** The expectation of the score function is zero, conditional on any present or past state. For any $0\le i\le j\le T$ and any state $s_i$,

$$
\mathbb E_{\pi_\theta}\left[\nabla_\theta\ln\pi_\theta(A_j\mid S_j)\;\Big|\;S_i=s_i\right]=0.
$$

Further, if $\Psi_i$ is a random variable independent of $A_i,S_{i+1},A_{i+1},\dots$, then

$$
\mathbb E_{\pi_\theta}\left[\nabla_\theta\ln\pi_\theta(A_j\mid S_j)\cdot\Psi_i\;\Big|\;S_i=s_i\right]=0.
$$

**Proof sketch (as stated):** use the log-derivative trick (score function trick).

#### Proof of the two identities (as stated)

Applying the log-derivative trick:

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\nabla_\theta\,\mathbb E_{\pi_\theta}\left[\sum_{i\in 0:T}\gamma^i R_i\;\Big|\;S_0=s_0\right]\\
&=\mathbb E_{\pi_\theta}\left[\left(\sum_{i\in 0:T}\gamma^i R_i\right)\nabla_\theta\ln\big(\pi_\theta(A_0,\dots,A_T\mid S_0,\dots,S_T)\big)\;\Big|\;S_0=s_0\right]\\
&=\mathbb E_{\pi_\theta}\left[\left(\sum_{i\in 0:T}\gamma^i R_i\right)\sum_{j\in 0:T}\nabla_\theta\ln\big(\pi_\theta(A_j\mid S_j)\big)\;\Big|\;S_0=s_0\right]\\
&=\mathbb E_{\pi_\theta}\left[\sum_{i,j\in 0:T}(\gamma^i R_i)\,\nabla_\theta\ln\pi_\theta(A_j\mid S_j)\;\Big|\;S_0=s_0\right].
\end{aligned}
$$

By the lemma, for any $0\le i<j\le T$,

$$
\mathbb E_{\pi_\theta}\left[(\gamma^i R_i)\,\nabla_\theta\ln\pi_\theta(A_j\mid S_j)\;\Big|\;S_0=s_0\right]=0,
$$

so the expression reduces to

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\mathbb E_{\pi_\theta}\left[\sum_{0\le j\le i\le T}(\gamma^i R_i)\,\nabla_\theta\ln\pi_\theta(A_j\mid S_j)\;\Big|\;S_0=s_0\right]\\
&=\mathbb E_{\pi_\theta}\left[\sum_{j\in 0:T}\nabla_\theta\ln\pi_\theta(A_j\mid S_j)\;\sum_{i\in j:T}(\gamma^i R_i)\;\Big|\;S_0=s_0\right].
\end{aligned}
$$

Thus, an **unbiased estimator** is

$$
\nabla_\theta J(\theta)\approx \frac1N\sum_{n=1}^N\left[\sum_{t\in 0:T}\nabla_\theta\ln\pi_\theta(A_{t,n}\mid S_{t,n})\;\sum_{\tau\in t:T}(\gamma^{\tau-t}R_{\tau,n})\right],
$$

where $n$ ranges over $N$ rollout trajectories from $\pi_\theta$.

The score function $\nabla_\theta\ln\pi_\theta(A_t\mid S_t)$ can be interpreted as the direction in parameter space that increases the probability of taking action $A_t$ in state $S_t$.

### Algorithm

REINFORCE is a loop:

1. Roll out $N$ trajectories using policy $\pi_{\theta_t}$.
2. Compute the gradient estimate

   $$
   g_i\leftarrow \frac1N\sum_{n=1}^N\left[\sum_{t\in 0:T}\nabla_{\theta_t}\ln\pi_\theta(A_{t,n}\mid S_{t,n})\sum_{\tau\in t:T}(\gamma^\tau R_{\tau,n})\right].
   $$

3. Update by gradient ascent

   $$
   \theta_{i+1}\leftarrow \theta_i+\alpha_i g_i,
   $$

   where $\alpha_i$ is the learning rate at step $i$.

---

## Variance reduction

REINFORCE is **on-policy**, meaning trajectories used for the update must be sampled from the current policy $\pi_\theta$. This can lead to high variance because returns $R(\tau)$ can vary significantly across trajectories. Many variants are introduced under **variance reduction**.

### REINFORCE with baseline

A common variance-reduction identity is

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t\in 0:T}\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\left(\sum_{\tau\in t:T}(\gamma^\tau R_\tau)-b(S_t)\right)\;\Big|\;S_0=s_0\right],
$$

for any function $b:\text{States}\to\mathbb R$.

The modified estimator is

$$
g_i\leftarrow \frac1N\sum_{n=1}^N\left[\sum_{t\in 0:T}\nabla_{\theta_t}\ln\pi_\theta(A_{t,n}\mid S_{t,n})\left(\sum_{\tau\in t:T}(\gamma^\tau R_{\tau,n})-b_i(S_{t,n})\right)\right],
$$

and REINFORCE is the special case $b_i\equiv 0$.

### Actor-critic methods

If the baseline is chosen so that

$$
b_i(S_t)\approx \mathbb E\left[\sum_{\tau\in t:T}(\gamma^\tau R_\tau)\;\Big|\;S_t\right]=\gamma^t V^{\pi_{\theta_i}}(S_t),
$$

variance can be significantly decreased, approaching

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t\in 0:T}\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\left(\sum_{\tau\in t:T}(\gamma^\tau R_\tau)-\gamma^t V^{\pi_\theta}(S_t)\right)\;\Big|\;S_0=s_0\right].
$$

In **actor-critic** methods, the policy is the actor and a learned value function is the critic.

The **Q-function** can also be used as critic:

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{0\le t\le T}\gamma^t\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\cdot Q^{\pi_\theta}(S_t,A_t)\;\Big|\;S_0=s_0\right].
$$

Subtracting $V^\pi$ yields the **advantage**

$$
A^\pi(S,A)=Q^\pi(S,A)-V^\pi(S),
$$

and an equivalent form:

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{0\le t\le T}\gamma^t\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\cdot A^{\pi_\theta}(S_t,A_t)\;\Big|\;S_0=s_0\right].
$$

In summary, many unbiased estimators take the form

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{0\le t\le T}\nabla_\theta\ln\pi_\theta(A_t\mid S_t)\cdot\Psi_t\;\Big|\;S_0=s_0\right],
$$

where $\Psi_t$ is any linear sum of the following terms:

- $\sum_{0\le \tau\le T}(\gamma^\tau R_\tau)$ (never used).
- $\gamma^t\sum_{t\le \tau\le T}(\gamma^{\tau-t}R_\tau)$ (REINFORCE).
- $\gamma^t\sum_{t\le \tau\le T}(\gamma^{\tau-t}R_\tau)-b(S_t)$ (REINFORCE with baseline).
- $\gamma^t\left(R_t+\gamma V^{\pi_\theta}(S_{t+1})-V^{\pi_\theta}(S_t)\right)$ (1-step TD).
- $\gamma^t Q^{\pi_\theta}(S_t,A_t)$.
- $\gamma^t A^{\pi_\theta}(S_t,A_t)$.

Additional examples (similar proofs):

- $\gamma^t\left(R_t+\gamma R_{t+1}+\gamma^2 V^{\pi_\theta}(S_{t+2})-V^{\pi_\theta}(S_t)\right)$ (2-step TD).
- $\gamma^t\left(\sum_{k=0}^{n-1}\gamma^k R_{t+k}+\gamma^n V^{\pi_\theta}(S_{t+n})-V^{\pi_\theta}(S_t)\right)$ (n-step TD).
- $\gamma^t\sum_{n=1}^\infty \frac{\lambda^{n-1}}{1-\lambda}\cdot\left(\sum_{k=0}^{n-1}\gamma^k R_{t+k}+\gamma^n V^{\pi_\theta}(S_{t+n})-V^{\pi_\theta}(S_t)\right)$ (TD($\lambda$), **GAE**).

---

## Natural policy gradient

The **natural policy gradient** method (Kakade, 2001) aims to provide a coordinate-free update (unlike standard policy gradient updates, which depend on the choice of parameters $\theta$).

### Motivation

Standard updates:

$$
\theta_{i+1}=\theta_i+\alpha\nabla_\theta J(\theta_i)
$$

solve:

$$
\begin{cases}
\max_{\theta_{i+1}}\; J(\theta_i)+(\theta_{i+1}-\theta_i)^T\nabla_\theta J(\theta_i)\\
\|\theta_{i+1}-\theta_i\|\le \alpha\,\|\nabla_\theta J(\theta_i)\|
\end{cases}
$$

The Euclidean constraint introduces coordinate dependence, so replace it with a KL constraint:

$$
\begin{cases}
\max_{\theta_{i+1}}\; J(\theta_i)+(\theta_{i+1}-\theta_i)^T\nabla_\theta J(\theta_i)\\
\bar D_{\mathrm{KL}}(\pi_{\theta_{i+1}}\|\pi_{\theta_i})\le \epsilon
\end{cases}
$$

with average KL divergence

$$
\bar D_{\mathrm{KL}}(\pi_{\theta_{i+1}}\|\pi_{\theta_i}) := \mathbb E_{s\sim \pi_{\theta_i}}\left[D_{\mathrm{KL}}\big(\pi_{\theta_{i+1}}(\cdot\mid s)\|\pi_{\theta_i}(\cdot\mid s)\big)\right].
$$

### Fisher information approximation

For small $\epsilon$:

$$
\bar D_{\mathrm{KL}}(\pi_{\theta_{i+1}}\|\pi_{\theta_i})\approx \frac12(\theta_{i+1}-\theta_i)^T F(\theta_i)(\theta_{i+1}-\theta_i),
$$

where the Fisher information matrix is

$$
F(\theta)=\mathbb E_{s,a\sim \pi_\theta}\left[\nabla_\theta\ln\pi_\theta(a\mid s)\,\big(\nabla_\theta\ln\pi_\theta(a\mid s)\big)^T\right].
$$

This yields the natural policy gradient update:

$$
\theta_{i+1}=\theta_i+\alpha F(\theta_i)^{-1}\nabla_\theta J(\theta_i),
$$

with step size typically set to maintain the KL constraint:

$$
\alpha\approx \sqrt{\frac{2\epsilon}{\big(\nabla_\theta J(\theta_i)\big)^T F(\theta_i)^{-1}\nabla_\theta J(\theta_i)}}.
$$

Inverting $F(\theta)$ is computationally intensive in high dimensions; practical implementations use approximations.

---

## Trust Region Policy Optimization (TRPO)

**TRPO** extends natural policy gradient by enforcing a trust-region constraint on policy updates (Schulman et al., 2015). It uses a line search and KL constraint to keep updates within a region where the approximations hold.

### Formulation

TRPO solves:

$$
\begin{cases}
\max_\theta\; L(\theta,\theta_i)\\
\bar D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta_i})\le \epsilon
\end{cases}
$$

where the surrogate advantage is

$$
L(\theta,\theta_i)=\mathbb E_{s,a\sim \pi_{\theta_i}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_i}(a\mid s)}\,A^{\pi_{\theta_i}}(s,a)\right].
$$

More generally:

$$
L(\theta,\theta_i)=\mathbb E_{s,a\sim \pi_{\theta_i}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_i}(a\mid s)}\,\Psi^{\pi_{\theta_i}}(s,a)\right].
$$

At $\theta=\theta_t$,

$$
\nabla_\theta L(\theta,\theta_t)=\nabla_\theta J(\theta)=\mathbb E_{(s,a)\sim \pi_\theta}\left[\nabla_\theta\ln\pi_\theta(a\mid s)\cdot A^{\pi_\theta}(s,a)\right].
$$

With Taylor approximations around $\theta_i$:

$$
\begin{aligned}
L(\theta,\theta_i)&\approx g^T(\theta-\theta_i),\\
\bar D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta_i})&\approx \frac12(\theta-\theta_i)^T H(\theta-\theta_i),
\end{aligned}
$$

where $g=\nabla_\theta L(\theta,\theta_i)\rvert_{\theta=\theta_i}$ and $F=\nabla_\theta^2\bar D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta_i})\rvert_{\theta=\theta_i}$ (Fisher matrix). This yields:

$$
\theta_{i+1}=\theta_i+\sqrt{\frac{2\epsilon}{g^T F^{-1}g}}\,F^{-1}g.
$$

TRPO then:

- uses conjugate gradient to solve $Fx=g$ without explicit inversion;
- uses backtracking line search to satisfy KL constraint and ensure improvement (testing $\theta_i+\alpha^k\sqrt{\tfrac{2\epsilon}{x^T Fx}}x$ for $k=0,1,2,\dots$).

---

## Proximal Policy Optimization (PPO)

**PPO** avoids computing $F(\theta)$ and $F(\theta)^{-1}$ by using clipped probability ratios.

Instead of constrained maximization, PPO uses the clipped objective:

$$
\max_\theta\;\mathbb E_{s,a\sim \pi_{\theta_t}}\left[
\begin{cases}
\min\left(\frac{\pi_\theta(a\mid s)}{\pi_{\theta_t}(a\mid s)},\,1+\epsilon\right)A^{\pi_{\theta_t}}(s,a) & \text{if }A^{\pi_{\theta_t}}(s,a)>0,\\
\max\left(\frac{\pi_\theta(a\mid s)}{\pi_{\theta_t}(a\mid s)},\,1-\epsilon\right)A^{\pi_{\theta_t}}(s,a) & \text{if }A^{\pi_{\theta_t}}(s,a)<0.
\end{cases}
\right].
$$

PPO performs multiple optimization steps on the same batch, keeping $\theta$ proximal to $\theta_t$ to remain effectively on-policy.

If a reference policy $\pi_{\text{ref}}$ is used, an additional KL penalty may be added:

$$
-\beta\,\mathbb E_{s,a\sim \pi_{\theta_t}}\left[\log\left(\frac{\pi_\theta(a\mid s)}{\pi_{\text{ref}}(a\mid s)}\right)\right],
$$

or equivalently:

$$
-\beta\,\mathbb E_{s,a\sim \pi_{\theta_t}}\left[\log\left(\frac{\pi_\theta(a\mid s)}{\pi_{\text{ref}}(a\mid s)}\right)+\frac{\pi_{\text{ref}}(a\mid s)}{\pi_\theta(a\mid s)}-1\right].
$$

### Group Relative Policy Optimization (GRPO)

**GRPO** is a PPO variant that omits the value function estimator $V$. For each state $s$, sample $G$ actions $a_1,\dots,a_G\sim \pi_{\theta_t}$ and compute the group-relative advantage

$$
A^{\pi_{\theta_t}}(s,a_j)=\frac{r(s,a_j)-\mu}{\sigma},
$$

where $\mu,\sigma$ are the mean and standard deviation of $r(s,a_1),\dots,r(s,a_G)$. Then maximize the PPO objective averaged across actions:

$$
\max_\theta\;\frac1G\sum_{i=1}^G\mathbb E_{(s,a_1,\dots,a_G)\sim \pi_{\theta_t}}\left[
\begin{cases}
\min\left(\frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_t}(a_i\mid s)},\,1+\epsilon\right)A^{\pi_{\theta_t}}(s,a_i) & \text{if }A^{\pi_{\theta_t}}(s,a_i)>0,\\
\max\left(\frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_t}(a_i\mid s)},\,1-\epsilon\right)A^{\pi_{\theta_t}}(s,a_i) & \text{if }A^{\pi_{\theta_t}}(s,a_i)<0.
\end{cases}
\right].
$$

---

## Policy Optimization and the Mirror Descent perspective (MDPO)

TRPO, PPO, and natural policy gradient share the idea of updating along the policy gradient while keeping updates stable via a distance to the previous policy.

Mirror Descent (proximal optimization) updates $\mathbf x_t$ via

$$
\mathbf x_{t+1}\in\arg\min_{\mathbf x\in\mathcal C}\;\nabla f(\mathbf x_t)^T(\mathbf x-\mathbf x_t)+\frac1{\eta_t}B_\omega(\mathbf x,\mathbf x_t).
$$

This motivates MDPO. With KL as the Bregman divergence:

$$
\pi_{t+1}\in\arg\max_\pi\;\mathbb E_{s,a\sim \pi}\big[A^{\pi_t}(s,a)\big]-\frac1{\eta_t}D_{\mathrm{KL}}(\pi\|\pi_t).
$$

With parameterized policy $\pi_\theta$:

$$
\max_\theta\;L(\theta,\theta_t)=\mathbb E_{s,a\sim \pi_{\theta_t}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_t}(a\mid s)}A^{\pi_{\theta_t}}(s,a)\right]-\frac1{\eta_t}D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta_t}).
$$

This objective can be used with techniques like PPO clipping; the KL penalty also appears in the original PPO paper.

## See also

- Reinforcement learning
- Deep reinforcement learning
- Actor-critic method

## References

1. Sutton et al. (1999) _Policy Gradient Methods for Reinforcement Learning with Function Approximation_.
2. Mohamed et al. (2020) _Monte Carlo Gradient Estimation in Machine Learning_.
3. Williams (1992) _Simple statistical gradient-following algorithms for connectionist reinforcement learning_.
4. Schulman et al. (2018) _High-Dimensional Continuous Control Using Generalized Advantage Estimation_.
5. Kakade (2001) _A Natural Policy Gradient_.
6. Schulman et al. (2015) _Trust region policy optimization_.
7. Schulman et al. (2017) _Proximal Policy Optimization Algorithms_.
8. Stiennon et al. (2020) _Learning to summarize with human feedback_.
9. Shao et al. (2024) _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_.
10. Nemirovsky & Yudin (1983) _Problem Complexity and Method Efficiency in Optimization_.
11. Shani et al. (2020) _Adaptive Trust Region Policy Optimization_.
12. Tomar et al. (2020) _Mirror Descent Policy Optimization_.
