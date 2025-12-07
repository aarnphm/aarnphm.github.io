---
date: "2024-11-27"
description: three-dimensional generalization of leibniz integral rule for differentiating integrals over time-dependent regions with moving boundaries.
id: Reynolds transport theorem
modified: 2025-10-29 02:15:33 GMT-04:00
tags:
  - math
  - calculus
title: Reynolds transport theorem
---

Also known as _Leibniz-Reynolds transport therem_

A three-dimensional generalization of Leibniz integral rule

> [!math] theorem
>
> Consider integrating $\mathbf{f} = \mathbf{f}(x, t)$ over time-dependent region $\Omega (t)$ that has boundary $\partial \Omega (t)$ then take derivative w.r.t time:
>
> $$
> \frac{d}{dt} \int_{\Omega (t)} \mathbf{f} dV
> $$

## general form

$$
\frac{d}{dt} \int_{\Omega(t)} \mathbf{f} dV = \int_{\Omega (t)} \frac{\partial{\mathbf{f}}}{\partial{t}} dV + \int_{\partial{\Omega (t)}}(\mathbf{v}_b \cdot \mathbf{n}) \mathbf{f} dA
$$

where:

- $\mathbf{n}(\mathbf{x},t)$ is the outward-pointing unit normal vector
- $\mathbf{x}$ is the variable of integrations
- $dV$ and $dA$ are volume and surface elements at $\mathbf{x}$
- $\mathbf{v}_b(\mathbf{x},t)$ is the velocity of the area element.
