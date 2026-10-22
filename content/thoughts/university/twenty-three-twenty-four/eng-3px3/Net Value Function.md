---
date: '2024-01-09'
description: net value as benefits minus costs, measured against a stated alternative and from a stated perspective; client, producer and system net value, common units, and marginal value.
id: Net Value Function
modified: 2026-10-22 11:05:00 GMT-04:00
tags:
  - eng3px3
title: Net value function
---

See [[thoughts/university/twenty-three-twenty-four/eng-3px3/3PX3 Net Value Functions.pdf|slides]]

## economics versus accounting

Economics studies how to allocate scarce resources between competing uses. Every choice has a cost, because the resources it uses are no longer available for the next best option.

The slides use a simplified accounting profit, sales revenue minus the listed production costs. Revenue and expenses can be recognized before cash changes hands:

$$
\text{Profit} = \text{Sales revenue} - \text{Labour} - \text{Materials} - \text{Factory production}
$$

In the slides' speedboat example, option C has the largest profit. Profit does not show whether C is riskier, pollutes more, or creates social inequities. Engineering economics puts these effects into the same comparison, so that options can be ranked on one quantitative scale.

Cost-benefit analysis starts with _Jules Dupuit_, an engineer who in 1844 asked how to measure the usefulness of public works such as bridges and roads. See [Economics evolving: a history of economic thought by _Agnar Sandmo_](https://press.princeton.edu/books/paperback/9780691148427/economics-evolving).

## net value

The net value of a project is the value of its benefits minus the cost of producing it:

$$
\text{Net value} = \text{Benefits} - \text{Costs}
$$

A net value has meaning only when two things are stated:

1. **The baseline.** Net value is relative: relative to doing nothing, or relative to another option.
2. **The perspective.** Benefits to one party can be costs to another.

For a sale between a client and a producer, the sale price is a transfer. It is a cost to the client and a benefit to the producer:

$$
\begin{aligned}
\text{NV}_{\text{client}} &= \text{Benefits}_{\text{client}} - \text{Sale price} \\
\text{NV}_{\text{producer}} &= \text{Sale price} - \text{Cost}_{\text{producer}}
\end{aligned}
$$

Both parties gain from the sale when both net values are positive:

$$
\text{Benefits}_{\text{client}} > \text{Sale price} > \text{Cost}_{\text{producer}}
$$

Add the two net values and the price cancels:

$$
\text{System net value} = \text{NV}_{\text{client}} + \text{NV}_{\text{producer}} = \text{Benefits}_{\text{client}} - \text{Cost}_{\text{producer}}
$$

The price decides how the gain is divided. The system net value decides how large the gain is. A larger system net value leaves a wider range of prices at which both sides gain.

## net value functions

A net value function (NVF) combines every factor of a scenario into one expression:

$$
\text{NVF} = \text{Benefit of solution} - \text{Cost of space} - \text{Cost of time} - \text{Cost of labour} - \cdots
$$

The course adds terms for environmental impact, ethical considerations and the time value of money. The inputs come from estimates and assumptions (mostly in 3PX3), from the client and from research (mostly in practice), and from engineering analysis, for example how heat transfer depends on flow rate.

### common units

The terms of an NVF must be in one unit before they can be added. Time, money, comfort and convenience are different units. The slides' example converts apples and oranges to dollars. Apples cost $3.89$ per kg at $100$ g each, and oranges cost $1.75$ per lb at $8$ oz each:

$$
\text{NV} = 5 \times \$0.389 - 3 \times \$0.875 = \$1.945 - \$2.625 = -\$0.68
$$

So by price alone, trading 3 oranges for 5 apples loses value. This result has three limits. The benefit to you is not equal to the market price. The time spent on the trade is not counted. And the value of each apple depends on how many you already have.

> [!notes] marginal value, quantity-dependent value
> The _marginal net value_ of one more apple is $\text{NV}(q+1)-\text{NV}(q)$. If quantity $q$ is treated as continuous, the corresponding slope is $\mathrm{d}\,\text{NV}/\mathrm{d}q$. Marginal net value is marginal benefit minus marginal cost. Diminishing benefit from successive apples gives decreasing net gains when marginal cost is constant.
>
> For a differentiable, concave NVF, a feasible interior point where marginal benefit equals marginal cost maximizes net value. A capacity limit can put the maximum at a boundary, so check those quantities too. For whole apples, compare feasible integer quantities. Stopping when the next apple's net gain becomes nonpositive works when successive net gains decrease. ^marginal-value

![[thoughts/university/twenty-three-twenty-four/eng-3px3/marginal-apple-q.webp]]
