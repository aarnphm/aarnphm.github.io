---
date: '2026-10-21'
description: Calendar-based tree growth, weighted species replacement, storms, and watering on the homepage.
id: forest years
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: forest years
---

`quartz/components/visitor-trail/forest-succession.ts` advances the forest by one simulated year per real week. Time starts at 15 October 2025 UTC. One real year therefore covers about $52$ forest years. A $220$-year pine lifespan corresponds to about $4.2$ real years; a $1500$-year sequoia lifespan corresponds to about $28.8$.

## species parameters

The table contains simulation parameters. Lifespan and maturity are measured in forest years. Maturity is the age at which the unwatered growth curve reaches about $90\%$ of full growth.

| Species | Lifespan | Maturity | Shade tolerance |
| ------- | -------: | -------: | --------------: |
| aspen   |     $70$ |     $18$ |          $0.10$ |
| birch   |     $80$ |     $22$ |          $0.20$ |
| willow  |     $60$ |     $14$ |          $0.20$ |
| larch   |    $160$ |     $35$ |          $0.15$ |
| pine    |    $220$ |     $45$ |          $0.35$ |
| oak     |    $320$ |     $60$ |          $0.45$ |
| sequoia |   $1500$ |    $120$ |          $0.40$ |
| cypress |    $300$ |     $55$ |          $0.60$ |
| spruce  |    $250$ |     $50$ |          $0.70$ |
| fir     |    $160$ |     $40$ |          $0.85$ |
| maple   |    $260$ |     $45$ |          $0.90$ |
| hemlock |    $400$ |     $70$ |          $0.95$ |

Each living tree receives a lifespan multiplier between $0.75$ and $1.25$. A founding snag is already dead at the census.

## replacement and storms

The source calls the species selection Horn's replacement rule. Small canopy gaps favour shade-tolerant species; open ground favours pioneers. For shade tolerance $s$, the selection weight is

$$
\lambda = \begin{cases} 1.3 - s & \text{open ground} \\ 0.25 + s & \text{a single-tree gap}. \end{cases}
$$

In a shaded gap, the previous species receives another factor of $s$. A hemlock replacing itself has weight $(0.25+0.95)\cdot0.95=1.14$. Aspen has weight $0.35$, reduced to $0.035$ when replacing itself. On open ground, aspen receives $1.2$ and hemlock $0.35$. These are relative weights in the candidate pool; normalisation determines the selection probabilities.

Each grove has storm probability $1/110$ per forest year. The mean interval is $110$ forest years, or about $2.1$ real years. A storm can remove a living tree once its age exceeds $0.4$ times its maturity age, with probability $0.75$. Younger trees survive.

## growth, care, and decay

For age $a\ge0$ and maturity age $m$, the unwatered growth fraction is

$$
h(a)=1-\exp\left(\frac{-2.3a}{m}\right).
$$

At $a=m$, $1-e^{-2.3}\approx0.9$. For $n$ waterings, care is $c=1-e^{-n/8}$. Care multiplies lifespan by $1+0.3c$ and adds an effective age of $0.4mc$ to the growth curve.

After natural death, broadleaf snags stand for $5$–$23$ forest years; conifer snags stand for $12$–$30$. Storm-felled trees fall immediately. Fallen broadleaf trunks remain for $12$–$32$ years and conifer trunks for $22$–$42$.

The next tree is born when its predecessor falls. A slot retains its identity and watering history across replacements, with a maximum of $64$ generations evaluated.
