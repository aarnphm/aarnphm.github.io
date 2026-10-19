---
date: '2024-02-01'
description: marginal analysis and sensitivity analysis techniques for evaluating net value function changes with respect to input parameters and conversion factors.
id: Sensitivity analysis
modified: 2026-06-05 15:08:34 GMT-04:00
tags:
  - eng3px3
title: Sensitivity analysis
---

See [[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 06 Sensitivity Analysis.pdf|slides]]

### Marginal analysis

Marginal analysis asks how net value changes when a decision changes by a small amount. If $V(q)$ is the net value of $q$ oranges, the extra value from one more orange is

$$
\Delta V=V(q+1)-V(q).
$$

This is a finite difference. When a continuous model is differentiable, its derivative gives the rate of change at a point:

$$
\frac{dV}{dq}=\lim_{\Delta q\to0}\frac{V(q+\Delta q)-V(q)}{\Delta q}.
$$

The derivative has units of net value per orange. Multiplying it by a small quantity change approximates the value change; a whole-orange difference need not equal the derivative. For several inputs, a partial derivative holds the other inputs fixed.

### Sensitivity analysis

Sensitivity analysis asks how a model's outcome changes when its inputs change. In these slides, the outcome is the [[thoughts/university/twenty-three-twenty-four/eng-3px3/Net Value Function|net value function]] (NVF). An input may be a decision we control, such as pipe diameter, or an uncertain parameter, such as a cost estimate. Changing a decision tests possible actions; changing an uncertain parameter tests how much the conclusion depends on the estimate.

The slides use a **one-at-a-time** procedure:

1. Record the base-case inputs and net value.
2. Choose a plausible range for one input. Keep every other input at its base value.
3. Recalculate net value across that range. Check whether the preferred decision changes.
4. Restore the base case before changing another input.

A partial derivative describes local sensitivity. Evaluating a finite range can reveal curvature or a threshold that a derivative at the base case misses. The range is part of the result: an input with a wide tested range may produce a large value change even when its local slope is modest.

Raw derivatives also depend on units. A derivative measured per metre changes numerically when expressed per millimetre. For a nonzero base input $p_0$, the effect of a small fractional change $r$ is approximately

$$
\Delta V\approx\left.\frac{\partial V}{\partial p}\right|_{p_0}p_0r,
\qquad \Delta p=p_0r.
$$

This puts the output changes in the same net-value units when comparing equal percentage input changes. A zero base input needs an absolute change instead.

A **spider plot** traces net value against percentage change for each input. A **tornado plot** shows the low- and high-case net values for each input, usually sorted by the width of that output range. Both should state the base case and the tested ranges.

For example, slide 22 uses a base net value of about \$644.58 per week. A 20% change in the space-cost input changes weekly net value by about \$30.63; the same percentage change in ingredient cost changes it by \$2.00. The first effect is about fifteen times the second in this scenario. This ranking follows from the model, base case, and chosen ranges. It does not show which estimate is more uncertain.

The one-at-a-time procedure leaves interactions untested. If two uncertain inputs can change together, calculate joint scenarios as well. The slides also warn against extending a result beyond the tested range. A sensitivity range supplies no probabilities by itself, so its endpoints do not determine an expected net value.

When the question concerns an optimized decision, distinguish evaluating the old decision under changed inputs from solving again with the changed inputs. Re-optimization can change both the decision and its value. [[thoughts/university/twenty-three-twenty-four/eng-3px3/Linear Optimization#Sensitivity reports|LP sensitivity reports]] describe some of these changes within stated validity ranges.
