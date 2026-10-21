---
date: '2026-10-21'
description: Character-trigram matching with Dice similarity for the 404 page suggestions.
id: departures
modified: 2026-10-21 02:50:00 GMT-04:00
seealso:
  - '[[posts/exploring nature on the web|exploring nature on the web]]'
tags:
  - technical
  - sapling
title: departures
---

The departures board suggests up to three pages related to the missing URL. Home remains the final departure. `quartz/components/scripts/404.inline.ts` ranks candidates by character-trigram similarity and selects random pages if none exceeds the score threshold.

## normalisation and trigrams

The path is converted to lowercase, decomposed accents are removed, and a final `.htm` or `.html` suffix is removed. Characters outside ASCII letters, digits, and slashes become hyphens. Leading and trailing hyphens and slashes are stripped.

Each normalised string receives two leading spaces and one trailing space. A sliding window extracts every three-character substring into a set. For trigram sets $A$ and $B$, Dice similarity is

$$
\frac{2\,|A \cap B|}{|A| + |B|}.
$$

Repeated trigrams count once because the representation is a set.

## ranking

Each candidate receives the maximum of three scores:

| Query             | Candidate             | Weight |
| ----------------- | --------------------- | -----: |
| Full missing path | Full page slug        |    $1$ |
| Last path segment | Last slug segment     | $0.95$ |
| Last path segment | Normalised page title |  $0.9$ |

The board retains the three highest scores strictly above $0.3$. Tag pages, the 404 page, and protected pages are excluded.
