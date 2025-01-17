---
date: "2024-11-27"
id: Helmholtz decomposition
modified: 2025-10-29 02:15:24 GMT-04:00
tags:
  - math
title: Helmholtz decomposition
---

> certain differentiable vector fields can be resolved into sum of an _irrotational_ vector field and _solenoidal_ vector vield

> [!math] definition
>
> for a vector field $\mathbf{F} \in C^1 (V, \mathbb{R}^n)$ defined on a domain $V \subseteq \mathbb{R}^n$, a Helmholtz decomposition is a pair of vector fields
> $\mathbf{G} \in C^1 (V, \mathbb{R}^n)$ and $\mathbf{R} \in C^1 (V, \mathbb{R}^n)$ such that:
>
> $$
> \begin{aligned}
> \mathbf{F}(\mathbf{r}) &= \mathbf{G}(\mathbf{r}) + \mathbf{R}(\mathbf{r}) \\
> \mathbf{G}(\mathbf{r}) &= - \nabla \Phi (\mathbf{r}) \\
> \nabla \cdot \mathbf{R}(\mathbf{r}) &= 0
> \end{aligned}
> $$

Here $\Phi \in C^2(V, \mathbb{R})$ is a scalar potential, $\nabla \Phi$ is its gradient, and $\nabla \cdot \mathbf{R}$ is the [[thoughts/Vector calculus#divergence]] of the vector field $R$
