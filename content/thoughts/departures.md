---
date: '2026-10-21'
description: the 404 departures board scores nearby pages with Dice's coefficient on character trigrams.
id: departures
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: departures
---

The board under the clock in `quartz/components/scripts/404.inline.ts` offers three pages near the path that 404'd. Home stays pinned as the last departure. If nothing scores, the board wanders.

A string is padded with a space on each side, `` `  ${s} ` ``, and cut into every three characters. Two sets $A$ and $B$ score

$$
\frac{2\,|A \cap B|}{|A| + |B|},
$$

Dice's coefficient. The missing path is normalised: case folded, accents stripped, `.html` removed, and anything that is not a letter, a digit, or a slash turned into a hyphen.

Three comparisons are taken and the max is kept. The whole slug is weighted $1$. The last path segment is weighted $0.95$. The page title, normalised the same way, is weighted $0.9$. A score has to clear $0.3$. The top three pass.
