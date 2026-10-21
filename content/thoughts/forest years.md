---
date: '2026-10-21'
description: one forest year per week on the homepage, and the weighted gap rule the code names after Horn.
id: forest years
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: forest years
---

The homepage forest keeps a calendar in `quartz/components/visitor-trail/forest-succession.ts`. The census is 15 October 2025 UTC. One forest year is one real week, so a year of the stand is $52$ weeks of the wall clock. A pine entered at $220$ forest years is a little over four real years on the page. A sequoia at $1500$ is about $28.8$ real years.

## the stand

Shade tolerance runs from aspen at $0.1$ to hemlock at $0.95$. Lifespan and the age of nine-tenths height, in forest years:

| kind | span | mature | shade |
| --- | ---: | ---: | ---: |
| aspen | $70$ | $18$ | $0.10$ |
| birch | $80$ | $22$ | $0.20$ |
| willow | $60$ | $14$ | $0.20$ |
| larch | $160$ | $35$ | $0.15$ |
| pine | $220$ | $45$ | $0.35$ |
| oak | $320$ | $60$ | $0.45$ |
| sequoia | $1500$ | $120$ | $0.40$ |
| cypress | $300$ | $55$ | $0.60$ |
| spruce | $250$ | $50$ | $0.70$ |
| fir | $160$ | $40$ | $0.85$ |
| maple | $260$ | $45$ | $0.90$ |
| hemlock | $400$ | $70$ | $0.95$ |

A drawn lifespan is $0.75$ to $1.25$ of the table, except a founding snag, which is already dead.

## who fills a gap

The comment calls the draw Horn's replacement rule: a gap one tree wide goes to whoever was waiting in the shade, and open ground after a storm goes to the pioneers. The weights below are this draw. With shade tolerance $s$,

$$
\lambda = \begin{cases} 1.3 - s & \text{open ground} \\ 0.25 + s & \text{a single-tree gap} \end{cases}
$$

and the previous species, in a gap that was not opened, is multiplied by $s$ again. A hemlock replacing itself in shade gets $(0.25 + 0.95)\cdot 0.95 = 1.14$. An aspen in that same gap gets $0.35$, or $0.035$ if it is the tree that just fell. On open ground the aspen gets $1.2$ and the hemlock $0.35$.

A storm is rolled once per grove per forest year at $1/110$, so the mean wait is $110$ forest years, about two real years. A tree past $0.4$ of its mature age and not yet dead is taken with probability $0.75$. Seedlings bend and stay.

## growth

Nine tenths of the height arrives at the mature age, and $1 - e^{-2.3} \approx 0.9$, so

$$
h(\mathrm{age}) = 1 - \exp\left(\frac{-2.3 \cdot \mathrm{age}}{\mathrm{mature}}\right).
$$

Watering counts as care $1 - e^{-n/8}$, which adds up to $30\%$ of the lifespan and brings the growth curve forward by $0.4$ of the mature age. Broadleaf snags stand $5$ to $23$ years and rot in $12$ to $32$. Needled snags stand $12$ to $30$ and rot in $22$ to $42$. The next trunk is born when the last one falls, and the slot is followed for at most $64$ generations.
