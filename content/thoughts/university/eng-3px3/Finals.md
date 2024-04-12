---
id: Finals
tags:
  - eng3px3
date: "2024-04-12"
title: Economics as an engineer, a guide.
---

## samples.

29.a
30.a

---

## behavioural economics

invisible hand of the market: self-interest of individuals leads to the best outcome for society as a whole, in a free market economy, as rational actors are motivated by incentives.

## tax, incentives and depreciations

_income, corporate, property, sales_

personal income tax: progressive tax rate
corporate tax: flat tax rate, regardless of income level -> net income: subtracting expenses from gross income.

profit on investments will be tax. If yields loss, then offset the loss against the profits from another to pay less tax overall.

[[thoughts/university/eng-3px3/Optimization|optimization]] strategies: minimize liabilities, timing of expenditures -> incorporate into financial models, do sensitivity analysis

before-tax MARR: set MARR high enough to include taxes that need to be paid
- good for intial assessments and comparisons with other investments => for investment's gross profit
after-tax MARR: if tax is explicitly accounted for in the cash flows of the project, then MARR should be lower
- good for final investment decisions, as reflect the true net return.

$$
MARR_{\text{after-tax}} = MARR_{\text{before-tax}} \times (1 - \text{corporate tax rate})
$$

_incentives_: tax credits, tax reliefs, programs to encourage certain activities
_depreciation_: due to use-related physical loss, technological obsolescence, functional loss, market fluctuation.

> Deprecation is a non-cash expense, but reduces the taxable income of a business.
> Can deduct annually by spreading the cost of an asset over its useful life.

affects NPV (net present value), IRR (internal rate of return), and payback period calculation

_Market value_: actual value of the asset can be sold for, estimated
_Book value_: deprecated value of the asset, using a depreciation model
_Salvage value_: estimated value of the asset at the end of its useful life

> [!important] value calculations
> Depreciation in year $n$ $D(n)$ is the decline in book value over that year: $BV(n) = BV(n-1) - D(n)$
>
> Salvage value $SV$ is the book value at object's EOL: $SV = BV(N) = MV(0) - \sum_{n=1}^{N} D(n)$

> [!note] Straight-line depreciation
> spreads uniformly over useful life, SLD of a period $D_{\text{sl}}(n) = \frac{\text{Purchase price}-\text{Salvage value after N periods}}{\text{N periods of useful life}}$.
>
> book value at end of $n^{th}$ year: $BV_{\text{sl}}(n) = P - n \times \frac{P-S}{N}$

> [!note] Declining-balance depreciation
> different assets are classified into classes: $D_{\text{db}}(n) = BV_{\text{db}}(n-1) \times d (\text{depreciation rate})$, such that book value at the end of a period $BV_{\text{db}}(n)$ is $BV_{\text{db}}(n) = P(1-d)^n$
>
> given salvage value $S$ and period of useful life $N$, depreciation rate $d = 1 - \sqrt[N]{\frac{S}{P}}$

> [!note] Sum-of-years-digits depreciation
> $D_{\text{syd}}(n) = \frac{N-n+1}{\sum_{i=1}^{N} i} \times (P-S)$

> [!note] Unit of production depreciation
> $D_{\text{uop}}(n) = \frac{\text{units produced of period}}{\text{life in \# of units}} \times (P - S)$
>
> assumes a SLD but vs. # of units rather than time.
