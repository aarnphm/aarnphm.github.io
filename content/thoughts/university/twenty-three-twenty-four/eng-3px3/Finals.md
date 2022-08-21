---
date: "2024-04-12"
description: Economics for engineer, a guide.
id: Finals
modified: 2025-10-29 02:16:13 GMT-04:00
tags:
  - eng3px3
title: Economics for engineer, a guide.
---

## samples.

4.b
11.e
12.c
13.d
14.b
15.a
16.e
17.c
18.c
19.b
20.c
21.b
22.c
23.b
24.a
25.a
26.e
27.a
28.e
29.a
30.a

---

## [[thoughts/university/twenty-three-twenty-four/eng-3px3/Net Value Function|net value function]]

$$
\text{NVF} = \text{benefit} - \text{cost}
$$

## conversion factors

![[thoughts/university/twenty-three-twenty-four/eng-3px3/Conversion Factors#marginal value change.|Conversion Factors]]

## optimisation

![[thoughts/university/twenty-three-twenty-four/eng-3px3/Optimization#model-based|Optimization]]

Linear optimization:
![[thoughts/university/twenty-three-twenty-four/eng-3px3/Linear Optimization#^linops|Linear Optimization]]

## time value of money

### interest

Interest $I$ is the compensation for loaning money.

> [!important] interest rate
> $i = \frac{I}{P}$. Thus $F = P(1+i)$

> [!important] Simple interests
> $I_{\text{each}} = P \times \frac{i}{\text{year}}$, total interest $I = I_{\text{each}} \times N_{\text{year}}$
>
> $F_n = P(1 + ni)$

> [!important] Compound interests
> $F_n = P(1+i)^n$

> [!important] nominal interest rates
> $r$ is the equivalent yearly rate if interest is withdrawn so it doesn't compound. (i.e: $r=mi$ where $m$ is the number of compounding periods per year)

> [!important] effective annual interest rates
> $i_{\text{eff}} = (1 + \frac{r}{m})^m - 1$

> [!important] effective interest rates
> how much interest do you accrue after a year if nominal rate is 12%?
> $F=P(1+i)^m=P(1+\frac{r}{m})^m$

> [!important] continuous compounding
> $F = P e^{ry}$

### net present value

$$
\text{NPV} = \text{CF}_0 + \sum_{n=1}^{N}{\frac{\text{CF}_n}{(1+i)^n}}
$$

where $\text{CF}_0$ is the initial cash flow, $\text{CF}_n$ is the cash flow at the end of the $n^{th}$ period, $i$ is the _effective interest rate_

> [!important] discount rate
> Present value $PV = \frac{\text{CF}_t}{(1+r_d)^t}$, where $\text{CF}_t$ is cash flow happening in $t$ years in the future, and $r_d$ is the discount rate.
>
> sources: opportunity cost, inflation, risk, time preference, inflation, option premium

regular deposit: Future value $FV = A \sum_{k=0}^{n-1}(1+i)^k = A \frac{(1+i)^n - 1}{i}$ where $A$ is the monthly, or time period, deposit.

fraction of last payment that was interest was $\frac{i}{1+i}$, principal of the last payment is $A = F_{\text{last}}(1+i)$

> [!important] geometric series
>
> $$
> \sum_{k=0}^{n-1}r^k = \frac{1-r^n}{1-r}
> $$

### inflation

> [!important] real vs. nominal
> nominal value refers to actual cash flow at the time it hapens, real value refers to equivalent amount of value at reference time, converted using inflation rates.
>
> real dollar $R = \frac{\text{CF}_n}{(1+r_i)^n}$, where $\text{CF}_n$ is the nominal cash flow at time $n$, and $r_i$ is the effective yearly inflation rate.

> [!important] internal rate of return
> the discount rate that results in a NPV of zero (break-even scenario)
>
> $$
> \text{CF}_0 + \sum_{n=1}^{N}{\frac{\text{CF}_n}{(1+r_{\text{IRR}})^n}} = 0
> $$

> [!important] minimum acceptable rate of return
> a rate of return set by stakeholders that must be earned for a project to be accepted
>
> real vs. nominal MARR: real MARR is MARR if returns are calculated using real dollars, whereas nominal MARR is MARR if returns are calculated using nominal dollars.
>
> $\text{MARR}_{\text{real}} = \frac{1+\text{MARR}}{1+f} - 1$ where $f$ is the inflation rate

## risk management and stochastic modelling

> Convert to dollar/wk to base calculation on same unit

uncertainty, evaluating likeliness and potential impact, organize to risk matrix, determine expected impact, then propose mitigation strategies
![[thoughts/university/twenty-three-twenty-four/eng-3px3/most-critical-risk.webp]]

> [!important] expected impact
> the chance it happens multiplied by the impact it will have if it happens.
> $\text{E[NPV]} = \sum_{i}{\text{NPV}(x_i)p(x_i)}$
>
> Then use this to create necessary mitigation

### NPV with risk and uncertainty

> [!note] probability distribution
> $p(x)$ of a discrete random variable $x$: Normalization requires that $\sum_{i}{p(x_i)} = 1$
>
> PDF (probability density function) $f(x)$ of a continuous random variable $x$: Normalization requires that $\int{p(x)dx} = 1$

> [!important] expected value for calculating stochastic to deterministic
> of function $f(x)$ is $\text{E}[f] = \sum_{i}{f(x_i)p(x_i)}$ for discrete random variable $x$ with probability distribution $p(x)$
>
> of function $f(x)$ is $\text{E}[f] = \int_x{f(x)p(x)dx}$ for continuous random variable $x$ with PDF $p(x)$

> [!note] Normal distribution
> $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
>
> `NORM.DIST(x, mean, stddev, cumulative)`: cumulative is `1` for CDF, `0` for PDF
> `NORM.INV(RAND(), 0.5, 0.05)`: draw values from a normal distribution with mean 0.5 and stddev 0.05

### non-linear deterministic and stochastic models

mean value $\mu_{x}$ of a random variable $x$ is its own expected value $\text{E}[x]$, variance $\sigma^2_{x}$ is the expected value of the squared deviation from the mean $\text{E}[(x-\mu_x)^2]$, and stddev $\sigma_x$

> [!important] central limit theorem
> sample size becomes large enough, the distribution of the sample mean will be approximately normally distributed, regardless of the distribution of the population, using [[thoughts/Monte-Carlo|Monte-Carlo]] simulation

> Expected value of linear and nonlinear functions: suppose $x$ and $y$ are independent random variables with means $\mu_x$ and $\mu_y$, and variances $\sigma^2_x$ and $\sigma^2_y$, then $E[x^{2}] = \sigma_x^2 - \mu_x^2$, $E[xy] = \int \int xyp_xp_ydxdy=\int xp_xdx \int yp_ydy=\mu_x \mu_y$

Dealing with 12 months per year: saying outcomes over a year should be **normally distributed** (CLT), with a mean given by expected value of monthly outcome and stddev given stddev of outcome divided by square root of the # of rolls ($\sqrt{12}$)

---

## project management and CPM

- scope, cost, time to maximize quality

WBS (work breakdown structure): hierarchical decomposition of the total scope of work

CPM (critical path method): determine the longest path through the network, the critical path, and the shortest time to complete the project
![[thoughts/university/twenty-three-twenty-four/eng-3px3/cpm.webp|cpm.webp]]

crashing a project means using additional resources to shorten a specific task

## supply and demand

market equilibrium: where supply and demand curves intersect, quantity demanded equals quantity supplied.
shift to right: greater demand, higher price, higher quantity. shift to left: lower demand, lower price, lower quantity.
factors of production: land, labour, capital, entrepreneurship
determinants of demand:

- price: quantity demanded $Q_d$ falls when price $P$ rises and vice versa
- prices of related goods: substitutes and complements
  determinants of supply:
- price: quantity supplied $Q_s$ rises when price $P$ rises and vice versa
- factors of productions
- fiscal policies, taxes, regulation

> [!important] elasticity: how responsive quantity demanded or supplied is to a change in price.
> Surplus when $Q_s > Q_d$, shortage when $Q_s < Q_d$.
>
> Elasticity of demand: $E_d = \frac{\% \Delta Q_d}{\% \Delta P} = \frac{\mid \frac{P}{Q_D} \mid}{\mid \frac{dP}{dQ_D} \mid}$
>
> Elasticity of supply: $E_s = \frac{\% \Delta Q_s}{\% \Delta P} = \frac{\mid \frac{P}{Q_S} \mid}{\mid \frac{dP}{dQ_S} \mid}$
>
> higher slope corresponds to lower elasticity: inelastic, lower slope corresponds to higher elasticity: elastic

Demand elasticity: $E_D <1$ means if price increases by 5% then demand will decrease by less than 5%, inelastic. $E_D >1$ means if price increases by 5% then demand will decrease by more than 5%, elastic.

> [!important] taxes
> arbitrary lower the equilibrium quantity,
>
> price seen by consumers vs. suppliers changes depends on relative elasticities of demand and supply: more price change will end up on consumer side
>
> quantities change depends on total elasticities of demand and supply: more elastic means more quantity change.

> [!important] subsidies
> arbitrary increase the equilibrium quantity,
>
> price seen by consumers vs. suppliers changes depends on relative elasticities of demand and supply: more price change will end up on consumer side
>
> quantities change depends on total elasticities of demand and supply: more elastic means more quantity change.

## behavioural economics

invisible hand of the market: self-interest of individuals leads to the best outcome for society as a whole, in a free market economy, as rational actors are motivated by incentives.

perfect competition: wheat (control of price none, low barrier to entry, high # of producers, products are identical)
monopolistic competition: restaurants (control of price low, low barrier to entry, high # of producers, products are similar)
oligopoly: airlines (control of price high, high barrier to entry, few producers, products are similar)
monopoly: utilities (control of price high, high barrier to entry, one producer, unique product)

game theory, most notable [[thoughts/The Prisoner's Dilemma|The Prisoner's Dilemma]]

anti-trust legislation: prevent monopolies, promote competition, protect consumers

> behavioural economics: + psychology to look at reasons people make _irrational_ decisions
>
> "bounded rationality": you don't have perfect information, and understand there's an opportunity cost to get it

law of demand and _ultimatum game_: people will pay less for a good if they can get it elsewhere for less, even if they value it more than the price they pay.

[[thoughts/Cooperation|Cooperation]]: R. Axelrod's _The Evolution of Cooperation_ propose a "strategy", what you do dependent on what the other person does.

PPF (production possibility frontier): trade-offs between two goods, given a fixed amount of resources.

risk aversion: people prefer a certain outcome to a risky one, even if the expected value of the risky one is higher. => assume that the given investment is loss, then calculate based on margin gains

## tax, incentives and depreciations

_income, corporate, property, sales_

personal income tax: progressive tax rate
corporate tax: flat tax rate, regardless of income level -> net income: subtracting expenses from gross income.

profit on investments will be tax. If yields loss, then offset the loss against the profits from another to pay less tax overall.

[[thoughts/university/twenty-three-twenty-four/eng-3px3/Optimization|optimization]] strategies: minimize liabilities, timing of expenditures -> incorporate into financial models, do sensitivity analysis

before-tax MARR: set MARR high enough to include taxes that need to be paid => for investment's gross profit
after-tax MARR: if tax is explicitly accounted for in the cash flows of the project, then MARR should be lower => for final investment decisions

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
