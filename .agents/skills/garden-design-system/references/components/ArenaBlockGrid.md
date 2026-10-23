# ArenaBlockGrid

A grid of Arena blocks that share 1px borders, under a lowercase mono section header.

**Source:** `arena.scss` (`.arena-section-header`, `.arena-channel-grid`, `.arena-block`, `.arena-block-title`).

## Inputs

- A section name.
- Per block: a title (clamped to three lines), an optional image or embed preview, and a meta line (source host, date). Each block opens the block view.

## Rules

- The grid draws its top and left edges and each cell draws its right and bottom, so every border is 1px.
- Columns: `auto-fill` with a 240px minimum on the site (140px in this preview).
- The pinned section's grid border is `secondary`.
- Hover: `gray` border. Focus: an inset `2px dark` ring.
- List mode stacks the same blocks in one column and shows the meta line.
