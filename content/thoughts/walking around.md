---
date: '2026-10-21'
description: the visibility graph and A* that walk the homepage's hikers and pet around trunks and camp solids.
id: walking around
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: walking around
---

Hikers, the patrol, and the pet detour in `quartz/components/visitor-trail/obstacle-route.ts`. A straight step is kept when `canMove` says the segment misses every solid. Otherwise the segment becomes a small visibility graph.

## the nodes

Endpoints that land inside a body are walked outward on rings of radius $4, 8, \ldots, 96$, sixteen angles, and the first free point is kept. For each obstacle near the segment, within a neighbourhood of $160$ on the fine pass and $80$ on the coarse one, the search drops a regular polygon around the expanded circle. The vertices have to sit outside that circle, and so do the chords between them. For a polygon of $n$ sides the circumradius is the inradius over $\cos(\pi/n)$:

$$
R = \frac{r_{\mathrm{body}} + r_{\mathrm{obstacle}} + p}{\cos(\pi/n)}.
$$

The fine pass uses $n = 32$ and padding $p = 2.25$. The coarse pass uses $n = 16$ and $p = 6$. A vertex that fails the free-point test is dropped.

## the search

Cost is Euclidean length. The heuristic is Euclidean distance to the goal, so the search is A* on a graph whose edges are straight and whose costs are the lengths of those edges. The heuristic is admissible. The implementation scans the open set linearly. An edge exists when `canMove` says the segment misses the blockers in the box around the nodes. Circles test the closest point on the segment against $(r + r_o + 2)^2$. Boxes test the sweep of the padded interval. From a point already inside a body, the test allows the move when the clearance is improving.

If the goal itself cannot be reached and a nearby arrival is allowed, the search keeps the visited node closest to that goal, inside $96$. The pet resamples the resulting polyline every $4$ pixels and carries scale and journey progress along the arc length, with its radius equal to its body size times the larger of the two endpoint scales.
