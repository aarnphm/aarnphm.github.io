# Queueing theory for engineers: Little's law, the hockey stick, and rate limiters

## (a) Core mental model

Three symbols carry the whole subject: arrival rate $\lambda$, service rate $\mu$ (per server), utilization $\rho = \lambda/\mu$ (for $c$ servers, $\rho = \lambda/(c\mu)$). Stability requires $\rho < 1$; at $\rho \ge 1$ the queue is a random walk with non-negative drift and $E[W]$ diverges (at exactly $\rho = 1$, M/M/1 is null-recurrent — it returns to empty, but the expected wait is still infinite).

The second load-bearing fact: **variability is the reason queues exist at all**. D/D/1 with $\rho < 1$ never queues — arrivals land every $1/\lambda$, each finishes $1/\mu < 1/\lambda$ later, the server is always idle when the next job arrives, waiting is identically zero at $\rho = 0.999$. Determinism kills waiting. Every real queue is therefore paying for exactly two things: utilization near 1, and variance in arrivals or service. Kingman's formula (below) makes that a product: delay $\approx$ variability $\times$ utilization $\times$ time. Everything else in this file is corollaries and the machinery to compute the constants.

A queue is an integrator of rate mismatch: $N(t)$ jumps up on bursts and drains at $\mu - \lambda$, so as $\rho \to 1$ the drain rate goes to zero and the integrator forgets nothing — that memory is the $1/(1-\rho)$ blow-up in every formula below.

## (b) Theory

### Little's law: $L = \lambda W$

Mean number in system $=$ arrival rate $\times$ mean time in system. **Distribution-free**: no Poisson assumption, no independence, any scheduling discipline, any boundary you can draw (a socket buffer, one service, a whole datacenter). Proof sketch: plot $N(t)$ over $[0, T]$ and measure the area under it two ways. Vertical strips: $\int_0^T N(t)\,dt = \bar{L} T$. Horizontal strips: each job contributes a strip of height 1 and length equal to its sojourn, so the same area is $\sum_i W_i = (\text{arrivals}) \cdot \bar{W} = \bar{\lambda} T \bar{W}$. Divide by $T$; done, up to edge effects that vanish as $T \to \infty$.

The three practical uses, with numbers:

1. **Sizing concurrency.** Target mean latency $W = 100$ ms at $\lambda = 500$ req/s $\Rightarrow$ $L = 50$ requests in flight. Your semaphore/connection pool/worker count must sustain $\approx 50$ concurrent, or $W$ was never achievable. This runs in reverse too: a pool of 64 at $\lambda = 500$/s hard-caps mean latency at $64/500 = 128$ ms _only if_ the pool never queues externally.
2. **Inferring in-flight from dashboards.** 2000 req/s at 50 ms mean latency $\Rightarrow$ 100 requests in flight right now, whether or not anything exports that gauge.
3. **Sanity-checking metrics.** Dashboard says $\lambda = 10{,}000$ req/s, mean latency 200 ms, concurrency gauge reads 500. Little demands $10{,}000 \times 0.2 = 2000$. One of the three numbers is a lie — the usual culprit is latency measured inside the handler while requests queue in the listen backlog, invisible to all three metrics.

Worked example end-to-end: a batch pipeline admits jobs at $\lambda = 4$/s and holds on average $L = 120$ jobs (queued + running). Then $W = L/\lambda = 30$ s per job regardless of scheduling policy, service distribution, or how many workers there are. If the SLO is 10 s, you need $L \le 40$ at this $\lambda$ — shed, scale, or slow admission; no scheduler shuffle can dodge the identity.

### Kendall notation and the Poisson process

$A/S/c/K$: interarrival distribution / service distribution / servers / capacity ($K$ omitted $=$ unbounded). $M$ means memoryless (exponential). The ones that matter: M/M/1, M/D/1, M/G/1, G/G/1, M/M/c.

Poisson process $=$ counts in disjoint intervals are independent, interarrivals are $\mathrm{Exp}(\lambda)$. Memorylessness: $P(T > s + t \mid T > s) = P(T > t)$ — having waited tells you nothing; the exponential is the only continuous distribution with this property (constant hazard rate). **Merging**: independent Poissons superpose to Poisson with $\lambda_1 + \lambda_2$. **Splitting**: independent coin-flip thinning with probability $p$ gives Poisson $p\lambda$. Bonus fact worth dropping: PASTA — Poisson arrivals see time averages, which is why formulas can equate "what an arrival experiences" with the steady-state distribution.

Where Poisson is realistic: many independent clients, none dominant — Palm–Khintchine says a superposition of many sparse independent renewal streams converges to Poisson. Ten thousand mobile clients, each occasional: Poisson-ish. Where it dies: **retry storms**. A timeout-plus-retry policy correlates arrivals with your own congestion state — the server slows, clients time out, each timed-out request respawns as $1 + r$ copies, so effective $\lambda$ multiplies by up to $1 + r$ precisely when capacity dropped. This is self-inflicted arrival-rate amplification and the core mechanism of metastable failure: the triggering slowness heals, the amplified load keeps the system pinned above $\rho = 1$ indefinitely. Cure: retry budgets, not retry counts — retries as a bounded fraction of fresh traffic (Finagle's default budget is 20%, the AWS SDKs run a client-side retry token bucket). Also non-Poisson: cron-aligned clients, cache-expiry thundering herds, anything synchronized by push.

### M/M/1: the formulas and the hockey stick

$$L = \frac{\rho}{1-\rho}, \quad W = \frac{1}{\mu - \lambda}, \quad L_q = \frac{\rho^2}{1-\rho}, \quad W_q = \frac{\rho}{\mu - \lambda}$$

Little links them: $L = \lambda W$, $L_q = \lambda W_q$, $L = L_q + \rho$. The form to memorize is $W = E[S]/(1-\rho)$ — sojourn in multiples of the service time:

| $\rho$ | $W$ in multiples of $E[S]$ |
| ------ | -------------------------- |
| 0.50   | 2                          |
| 0.80   | 5                          |
| 0.90   | 10                         |
| 0.95   | 20                         |
| 0.99   | 100                        |

Doubling load from 50% to 99% costs **50×** in latency. The curve is convex and everything interesting happens after 0.9 — which is why a load test that stops at 70% utilization ($W = 3.3\,E[S]$) predicts nothing about incidents: the distance from 0.7 to 0.95 is another 6×, and to 0.99 another 30×. The same table read backwards is the cheapest capacity argument you will ever make: at $\rho = 0.99$, adding 1% capacity ($\rho \to 0.98$) _halves_ latency.

### M/M/c and Erlang C

For $c$ servers sharing one queue, the Erlang C formula gives the probability an arrival waits, $C(c, \lambda/\mu)$, and $W_q = C \cdot E[S]/(c(1-\rho))$. You never compute it by hand; you remember what it implies: **pooling beats partitioning**. Ten separate M/M/1 lanes each at $\rho = 0.9$ have $W_q = 9\,E[S]$ per lane; one shared queue feeding the same 10 servers at the same total load has $C \approx 0.67$, so $W_q \approx 0.67\,E[S]$ — **13× better** from pooling alone, because a pooled system never has an idle server coexisting with a waiting job. The exceptions are real: cache locality (partitioned workers keep hot caches — per-core runqueues and prefix-cache-aware routing in LLM serving partition on purpose), fault/tenant isolation (blast radius), and head-of-line blocking across classes (one tenant's 10 s whale in the shared queue delays every other tenant; sometimes you partition precisely to contain that).

### M/G/1 and Pollaczek–Khinchine: service variance is first-class

$$W_q = \frac{\lambda E[S^2]}{2(1-\rho)} = \frac{\rho}{1-\rho} \cdot \frac{1 + C_s^2}{2} \cdot E[S], \qquad C_s^2 = \frac{\operatorname{Var}(S)}{E[S]^2}$$

$E[S^2]$, not $E[S]$, drives the queueing delay. Concrete: 99% of requests take 10 ms, 1% take 10 s. Then $E[S] = 110$ ms and $E[S^2] \approx 1.0$ s². At $\lambda = 5$/s, $\rho = 0.55$ — the server is nearly half idle — yet $W_q = 5 \times 1.0 / (2 \times 0.45) \approx 5.6$ s for _everyone_. Exponential service with the same mean would give $W_q \approx 0.13$ s: the 1% whales make the queue **41× worse** at identical utilization. One 10 s request among 10 ms requests wrecks every request behind it; means on a dashboard cannot see this, second moments can. Mitigations: SJF/SRPT (serve short/least-remaining first — provably optimal mean response time), size-based routing (whales get their own pool, back to the partitioning exception above), hedging/killing stragglers, and capping request cost at the API (pagination, timeouts) to truncate the $S$ distribution directly.

### Kingman's VUT approximation (G/G/1)

$$W_q \approx \frac{\rho}{1-\rho} \cdot \frac{C_a^2 + C_s^2}{2} \cdot E[S]$$

**V**ariability $\times$ **U**tilization $\times$ **T**ime — the back-of-envelope workhorse. With $C_a^2 = 1$ (Poisson arrivals) it coincides with P–K, so it is exact for M/G/1 and a heavy-traffic approximation otherwise. Both knobs are measurable from logs as squared coefficients of variation, which makes this the formula you actually use: batching upstream pushes $C_a^2$ from 1 to, say, 4; at $\rho = 0.9$ with $C_s^2 = 1$ that moves $W_q$ from $9\,E[S]$ to $22.5\,E[S]$ without a single extra request per second. It also gives the two levers ranked: near saturation the $\rho/(1-\rho)$ term dominates (buy capacity); at moderate load the variability term does (smooth arrivals, cap service tails).

### Scheduling and tails

FIFO is the baseline P–K analyzes: fair by arrival order, worst-case mean under high $C_s^2$. **SJF/SRPT** minimize mean response time (SRPT provably optimal); the starvation objection to elephants is weaker than folklore says — in heavy traffic even large jobs typically do better under SRPT than FIFO because the queue they eventually see is shorter. **Processor sharing** (what a CPU or an event loop approximates) is egalitarian and insensitive: M/G/1-PS mean response depends on $E[S]$ only, so PS neutralizes service variance without knowing job sizes. **LIFO** has terrible mean and variance, and one killer niche: under overload, LIFO-plus-shedding beats FIFO on tail SLOs. Reason: the newest request still has a live client waiting; the oldest has likely already timed out client-side, so FIFO burns capacity producing responses nobody reads (throughput without goodput). Facebook ships adaptive LIFO paired with CoDel-style shedding for exactly this. **Preemptive priority** gives the high class an M/M/1 that only sees its own load, while the low class divides by an extra $(1 - \rho_{\text{high}})$ factor — the formal version of "best-effort traffic starves at peak".

### Networks: Jackson and the bottleneck

A Jackson network (Poisson external arrivals, exponential servers, probabilistic routing) has a product-form stationary distribution: solve the traffic equations for each node's effective $\lambda_i$, then each node behaves like an independent M/M/1 at its own $\rho_i$ and the joint distribution is the product of the marginals. This is why per-stage $\rho$ is the first thing to compute in any pipeline — the stages decouple.

The bottleneck law needs even less: system throughput is capped at $1/D_{\max}$ where $D_{\max}$ is the largest per-job service demand at any stage, and as offered load rises, the bottleneck stage's queue absorbs _all_ the growth while every other stage stays flat. Find the stage whose $\rho$ hits 1 first; nothing else matters until it is fixed.

## Systems applications

### Rate limiting

- **Token bucket** $(r, B)$: bucket holds $\le B$ tokens, refills at $r$/s, a request costing $c$ passes iff $c$ tokens are available. Admits any traffic shaped "burst $\le B$, sustained $\le r$": over any interval $T$, admitted work $\le B + rT$. Implementation is the interview point: _lazy refill_ — store (tokens, last); on each request add $(now - last) \cdot r$, cap at $B$, subtract on admit. Two floats per key, no timers, no background threads. Ships in Linux `tc`, Envoy, AWS API throttling, Guava RateLimiter.
- **Leaky bucket as meter** equals a token bucket with different bookkeeping (GCRA, the theoretical-arrival-time formulation — one timestamp per key). **As queue** it is a different animal: requests enter a FIFO drained at exactly $r$ — a _shaper_ that adds delay and outputs perfectly paced traffic, versus the meter, a _policer_ that adds zero delay and passes bursts. nginx `limit_req` is the queue flavor (with `burst`/`nodelay` toggling toward meter behavior).
- **Sliding-window log**: store a timestamp per admitted request, count the trailing window exactly. Exact but $O(\text{limit})$ memory per key. **Sliding-window counter**: two counters per key; estimate the trailing count as $\text{prev} \cdot (1 - e/\text{window}) + \text{curr}$ where $e$ is the elapsed fraction of the current window. The approximation assumes the previous window's arrivals were uniform: bunched-at-the-end traffic can slip up to $\sim 2\times$ limit through a true trailing window (aligned windows stay exact at $\le$ limit). Cloudflare shipped this and measured 0.003% wrong decisions across 400M requests — the memory/accuracy trade at CDN key-cardinality. Log where exactness is money (billing, abuse); counter where keys number in the millions; token bucket where bursts are legitimate product behavior.

### Backpressure vs unbounded queues

An unbounded queue converts overload into unbounded latency; a bounded queue converts it into an explicit, policy-shaped failure — and only the second is debuggable. Bufferbloat is the network's version: fat FIFO buffers at $\rho \approx 1$ hold seconds of standing queue, so "no drops" means every packet waits. **CoDel**'s fix is choosing the right signal: not queue _length_ (meaningless across link rates) but per-packet _sojourn time_ — if the minimum sojourn over a 100 ms interval stays above the 5 ms target, drop from the head and tighten the interval. Sojourn is the SLO quantity, and head-drop signals the sender faster than tail-drop. In services the same ideas are: load shedding (reject at admission, before any expensive work — the cheapest point in the pipeline), and deadline propagation (each hop passes remaining budget downstream and drops work whose deadline already expired; a response computed after the caller gave up is pure waste that then amplifies retry load — gRPC deadlines exist for this).

### The tail at scale

Fan-out amplifies tails: $P(\text{any leg slow}) = 1 - p^n$ for $n$ parallel legs each fast with probability $p$. At per-leaf p99, a 100-leaf fan-out leaves $0.99^{100} \approx 37\%$ of requests untouched by a slow leg — 63% of user requests eat at least one leaf's p99, so the leaf's p99 is roughly the root's _median_. Countermeasures from Dean & Barroso: **hedged requests** — send a duplicate to another replica after the p95 delay, take the first answer; BigTable benchmark went p999 $1800 \to 74$ ms for ~2% extra load. **Tied requests** — enqueue on two replicas simultaneously, each cancels the other on dequeue, capturing most of the benefit with less duplicate work. Hedge above a high percentile and cap the hedge budget, or the hedges themselves become the retry storm.

### The power of two choices

Throw $n$ balls into $n$ bins uniformly: max load $\Theta(\log n / \log\log n)$ w.h.p. Sample **two** bins per ball and take the emptier: $\log\log n/\log 2 + \Theta(1)$ — an exponential improvement for one extra probe, and $d = 3$ adds almost nothing ($\log\log n / \log d$). At $n = 20{,}000$: max load ~7–8 for $d = 1$ vs 3 for $d = 2$ (run `two_choices` in this kit). This is P2C load balancing in Envoy and Finagle: full least-loaded scanning is expensive and, with stale load data, causes herding onto the momentarily-least-loaded instance; sampling two and picking the better gets nearly all the benefit with $O(1)$ probes and is robust to staleness.

### Open vs closed loop load generation, and coordinated omission

A **closed-loop** generator runs $k$ virtual users, each sending the next request only after the previous completes: arrival rate self-throttles as the system slows, so latency at saturation looks bounded ($k$ requests in flight is a Little's-law identity, not a finding). Production internet traffic is **open-loop**: arrivals keep coming at $\lambda$ regardless of how slow you got, and past saturation latency diverges — as it will in the incident. Benchmark open-loop unless production genuinely is a fixed worker pool. **Coordinated omission** (Tene): when the generator itself stalls behind the slow system, the requests it _would have sent_ never get recorded, so exactly the samples carrying the queueing delay vanish from the histogram and p999 reads 10–1000× too low. Fix: schedule intended send times up front and measure each response against its intended time, not its actual send (wrk2, HdrHistogram). If a benchmark reports p999 below the length of an observed stall, it is lying by omission.

## (c) Gotchas and interviewer follow-ups

1. **Utilization is not headroom.** "We run at 60%, we're fine" — $W$ is already $2.5\,E[S]$, and 60 → 95% is another 8×. The hockey stick is convex; averages of $\rho$ over a day hide the peak hour where the queue actually forms. Queues respond to instantaneous $\lambda$, not the daily mean.
2. **Little's law needs one boundary.** $L$, $\lambda$, $W$ must be measured over the same box. Mixing handler-only latency with edge arrival rate "violates" Little and sends you debugging phantoms; the law failing on real metrics means an unmeasured queue (listen backlog, kernel socket buffer, sidecar) is inside your boundary.
3. **Means lie under variance.** P–K says $W_q \propto E[S^2]$. Two services with identical mean latency and utilization can differ 40× in queueing delay. Ask for the service-time histogram, not the mean.
4. **Token bucket, cost > capacity:** admissible never — a request costing more than $B$ waits forever no matter how long the bucket refills. Reject it up front or split it; the "big batch request starves silently" bug ships constantly.
5. **Sliding-window counter is exact per aligned window, ~2× worst case per trailing window.** Say both halves; naming the failure mode (arrivals bunched at the previous window's end) is the difference between using it and understanding it.
6. **Retries are multiplicative, budget them.** A retry _count_ of 3 means 4× arrival amplification at exactly the moment $\mu$ dropped. A retry _budget_ (fraction of fresh traffic) caps the amplification by construction.
7. **The Lindley recursion is the whole FIFO simulator.** $W_0 = 0$, $W_i = \max(0, W_{i-1} + S_{i-1} - (A_i - A_{i-1}))$. Single pass, no event heap. Reaching for a priority queue on a single-server FIFO question signals you have not seen it.
8. **LIFO under overload is not unfair, it is goodput-aware.** Fairness to a client that already timed out is pure waste; FIFO under overload maximizes work on abandoned requests. Pair LIFO with shedding, cite adaptive LIFO + CoDel.
9. **Hedging feeds the storm if unbounded.** Hedge above p95 and cap total hedges at a few percent; hedging at p50 doubles load for noise.
10. **D/D/1 is the null hypothesis.** Whenever asked "why is there a queue at 40% utilization" the answer is variance ($C_a^2$ or $C_s^2$), not load — go straight to Kingman's numerator.
11. **Pooling has exceptions and they're askable.** Answer "one shared queue" first, then volunteer cache locality, isolation, and cross-tenant head-of-line blocking as the reasons real systems partition anyway.
12. **Closed-loop benchmarks + coordinated omission produce beautiful, fictional p999s.** If the load tool blocked while the server stalled, the histogram is missing exactly the samples that mattered.

## (d) Rapid-fire drills

1. M/M/1 at $\rho = 0.95$: mean sojourn in service-time multiples? → $20\,E[S]$.
2. $\lambda = 200$ req/s, mean latency 250 ms → in-flight? → $L = 50$.
3. At $\rho = 0.99$, capacity +1% does what to $W$? → halves it ($100 \to 50\,E[S]$).
4. M/D/1 vs M/M/1 queueing delay at the same $\rho$? → exactly half ($C_s^2 = 0$).
5. Erlang C computes what? → probability an arrival waits in M/M/c.
6. 10 lanes at $\rho = 0.9$ pooled into one M/M/10: $W_q$ improvement? → $\approx$ 13× ($9\,E[S] \to 0.67\,E[S]$).
7. Token bucket $r = 100$/s, $B = 500$, idle 5+ s: max instantaneous burst then sustained? → 500 at once, then 100/s.
8. Fan-out 50 legs, per-leaf p99: fraction of requests with zero slow legs? → $0.99^{50} \approx 61\%$.
9. Kingman with $C_a^2 = C_s^2 = 1$ reduces to? → M/M/1 $W_q$ exactly.
10. What breaks the Poisson-arrivals assumption in production? → retry storms (arrivals correlated with congestion), cron alignment, thundering herds.
11. Balls-into-bins $d=1$ vs $d=2$ max load? → $\Theta(\log n/\log\log n)$ vs $\log\log n/\log 2 + \Theta(1)$; at $n = 20$k, ~7 vs 3.
12. Coordinated omission in one sentence? → the generator stalls with the server, the unsent requests' latencies never enter the histogram, tails read fictionally low.
13. P–K: doubling $E[S^2]$ at fixed mean and $\lambda$ does what to $W_q$? → doubles it.
14. Sliding log vs sliding counter memory per key? → one timestamp per admitted request vs two integers.
15. Why can LIFO beat FIFO on tail SLO under overload? → newest requests still have live clients; FIFO serves the already-timed-out first (goodput vs throughput).
16. CoDel's congestion signal and constants? → per-packet sojourn time; 5 ms target over a 100 ms interval, head drop.
17. Little's-law identity in an M/M/1 answer sheet: $L$, $L_q$, $\rho$ relation? → $L = L_q + \rho$.
18. Why does a 70%-utilization load test predict nothing? → $W(0.7) = 3.3\,E[S]$ vs $W(0.99) = 100\,E[S]$; the convex tail after 0.9 is where incidents live.
