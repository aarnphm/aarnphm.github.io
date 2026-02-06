---
date: '2025-11-01'
description: evergreen hub for topology study anchored on munkres and mit 18.901.
id: topology
modified: 2025-12-02 16:43:46 GMT-05:00
socials:
  ocw: https://ocw.mit.edu/courses/18-901-introduction-to-topology-fall-2004/
tags:
  - math
  - topology
  - evergreen
title: topology
---

## roadmap

see also: {{sidenotes[munkres' topology]: james munkres, _topology_ (2nd ed.)}}

- **phase 0 — foundations**: set theory, logic, proof skills, and metric intuition (munkres ch. 1; see [[thoughts/norm|norm]] for metric-induced topology). pace: 1 week.
- **phase 1 — point-set core** (weeks 2–5): topological spaces, bases, subspaces, product and quotient topologies (munkres ch. 2–5). track with mit 18.901 weeks 1–3.
- **phase 2 — separation + countability** (weeks 6–7): separation axioms, urysohn lemma, metrization theorems (munkres ch. 16–22). mirror mit 18.901 weeks 4–5.
- **phase 3 — compactness + connectedness** (weeks 8–9): compactness, lindelöf, local compactness, connectedness (munkres ch. 26–31). align with mit 18.901 weeks 6–7 problem sets.
- **phase 4 — algebraic entrance** (weeks 10–12): fundamental group, covering spaces, van kampen (munkres ch. 51–56). follow mit 18.901 weeks 8–11 lectures.
- **phase 5 — beyond** (weeks 13+): transition into homology and homotopy (munkres appendix, hatcher ch. 0–1) before stepping into mit 18.906.

## mit alignment

> [!important] timeline keyed to fall 2025 mit 18.901 (tu/thu 11:00–12:30) with spring 2026 follow-up via 18.906.

- **18.901 problem sets**: ten sets mapped to roadmap phases. track mastery in [[thoughts/topology/mit-18-901-problemsets|18-901 problem sets]].
- **18.901 lecture notes**: ocw notes and recitations inform [[thoughts/topology/point-set|point-set]] and [[thoughts/topology/compactness|compactness]] subfiles.
- **18.906 preview**: schedule homology prerequisites during phase 5; log outcomes in [[thoughts/topology/algebraic-bridge|algebraic bridge]].

## subnotes

### year 1 (current)

- [[thoughts/topology/point set|point-set topology]] — definitions, bases, subspaces, product/quotient constructions, sample mit pset solutions.
- [[thoughts/topology/separation|separation]] — $t_0$ through normality, urysohn, tietze extension, metrization heuristics.
- [[thoughts/topology/compactness|compactness]] — compact, sequentially compact, lindelöf, local compactness, applications.
- [[thoughts/topology/fundamental group|fundamental group]] — loop spaces, van kampen, covering theory case studies.
- [[thoughts/topology/algebraic bridge|algebraic bridge]] — homology theory, chain complexes, stepping stones toward mit 18.906.
- [[thoughts/topology/simply connected|simple connectivity]] — why $\pi_1=0$ is the right condition, poincaré homology sphere counterexample.

### extended roadmap (years 2-4)

see [[thoughts/topology/poincare-roadmap|poincaré conjecture roadmap]] for comprehensive multi-year plan from point-set topology through perelman's proof.

- [[thoughts/topology/differential-foundations|differential foundations]] — smooth manifolds, morse theory (year 2).
- [[thoughts/topology/3-manifolds|3-manifold topology]] — heegaard, jsj, thurston geometries (year 2-3).
- [[thoughts/topology/ricci-flow|ricci flow]] — hamilton's program, perelman's breakthroughs (year 3-4).
- [[thoughts/topology/resources|resources]] — curated bibliography by phase.

## file plan

- maintain phased notes listed above in `content/thoughts/topology/`.
- aggregate 18.901 problem sets and solutions in `topology/mit-18-901-problemsets.md`.
- weave weekly reflections into stream entries referencing this evergreen hub.

## long-term goal: poincaré conjecture

beyond the 12-week roadmap, this study extends toward understanding perelman's proof of the poincaré conjecture. see [[thoughts/topology/poincare-roadmap|poincaré roadmap]] for:

- **the statement**: every simply connected closed 3-manifold is homeomorphic to $S^3$
- **why it matters**: characterizes 3-sphere uniquely via topology, solved one of millennium prize problems
- **proof strategy**: thurston geometrization via ricci flow with surgery (perelman 2002-2003)
- **prerequisite chain**: point-set → algebraic → differential → riemannian → geometric analysis → ricci flow
- **realistic timeline**: 3-4 years to proof comprehension, 5-7 years to internalization

key conceptual foundation: [[thoughts/topology/simply-connected|simple connectivity]] ($\pi_1=0$) is strictly stronger than $H_1=0$ (see poincaré homology sphere). this makes it the "right" topological condition.

## next actions

### immediate (weeks 10-12)

- draft [[thoughts/topology/point-set|point-set]] with munkres ch. 2–5 summary + mit pset 1 highlights.
- capture separation axioms in [[thoughts/topology/separation|separation]] and align with mit pset 3.
- assemble compactness case studies in [[thoughts/topology/compactness|compactness]] using mit pset 5 prompts.
- complete fundamental group phase with focus on [[thoughts/topology/simply-connected|simple connectivity]].

### spring semester (weeks 13+)

- transition to [[thoughts/topology/algebraic-bridge|homology theory]] via hatcher ch. 2.
- construct poincaré homology sphere and verify $H_1=0$ but $\pi_1 \neq 0$.
- prepare for mit 18.906 algebraic topology ii.
