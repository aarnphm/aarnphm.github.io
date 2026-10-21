---
date: '2026-10-21'
description: Visibility-graph construction and A* routing around obstacles on the homepage trail.
id: walking around
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: walking around
---

Hikers, the patrol, and the pet use `quartz/components/visitor-trail/obstacle-route.ts` to avoid trees and camp structures. A segment is kept when `canMove` accepts it. A blocked segment triggers a visibility-graph search.

## endpoints and candidate nodes

An endpoint inside an obstacle is moved to the first free sample on concentric rings. The search tests radii $4,8,\ldots,96$, with sixteen angles per ring.

Nearby obstacles contribute nodes on a regular polygon around their expanded collision circles. The polygon must keep its edges outside the circle. For $n$ sides, body radius $r_b$, obstacle radius $r_o$, and padding $p$, its circumradius is

$$
R = \frac{r_b + r_o + p}{\cos(\pi/n)}.
$$

The search tries a coarse pass first, then a fine pass if needed:

| Pass   | Sides $n$ | Padding $p$ | Neighbourhood beyond combined radii |
| ------ | --------: | ----------: | ----------------------------------: |
| Coarse |      $16$ |         $6$ |                                $80$ |
| Fine   |      $32$ |      $2.25$ |                               $160$ |

Candidates that fail the free-point test are discarded. All distances are in scene units.

## graph search

An edge connects two nodes when `canMove` accepts the segment between them. Edge cost is Euclidean length, and the heuristic is Euclidean distance to the goal. This gives A\* an admissible lower bound because a path through several edges cannot be shorter than the direct distance.

The implementation selects the next node by scanning the open set. Collision tests use obstacles within the bounding box of all candidate edges. For circular obstacles, the closest point on a segment is tested against $(r_b+r_o+2)^2$. Box obstacles use a padded swept-interval test. A move from inside an obstacle is allowed when it increases clearance.

If the final goal is unreachable and nearby arrival is enabled, the fine pass keeps the reachable visited node nearest the goal within $96$ units. Otherwise it returns failure.

The pet resamples the resulting polyline every four pixels. Scale and journey progress follow arc length. Routing uses body size multiplied by the larger endpoint scale as the collision radius.
