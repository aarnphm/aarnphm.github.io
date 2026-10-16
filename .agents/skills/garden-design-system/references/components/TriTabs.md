# TriTabs

A row of small mono tabs for maps, charts and best-effort categories.

**Source:** `triathlon.scss` (`.tri-map-tablist`, `.tri-map-tab`) and `triathlon-best-efforts.scss` (`.tri-be-cats`).

**Corrections:** in the source, rest labels are `gray`, the selected label is paper on `tri-accent` (3.31:1), and the shared rule hides focus on the selected tab.

## Inputs

- A `role="tablist"` of `role="tab"` buttons with `aria-selected`, roving tabindex and arrow keys, and the panel each tab controls.

## Rules

- `tri-tab` type (mono 0.62rem), padded `0.18rem 0.4rem`, 1px border of `gray` at 24%, square.
- Rest label: `darkgray`. Selected: `tri-accent` fill and border, `on-accent` label. A best-efforts category may fill with its own category colour.
- Every tab, selected or not, shows the `tri-be-focus` ring.
- The row may scroll sideways on its own. Pad it 2px so the focus ring is not clipped.
