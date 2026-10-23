# TriTabs

A row of small mono tabs for maps, charts, best-effort categories and the course workspace.

**Source:** `quartz/components/styles/triathlon/map-detail.scss` (`.tri-map-tablist`, `.tri-map-tab`, `.tri-map-tab-shortcut`, `.tri-map-tab-label`) and `triathlon-best-efforts.scss` (`.tri-be-cats`, `.tri-be-tabs`). The course workspace reuses the class in `course-workspace/Header.tsx`.

**Corrections:** in the source, the best-efforts tabs draw a 1px `tri-be-focus` ring, and the selected label is paper on `tri-accent` (3.31:1). This rendition uses the 2px `focus` ring and `on-accent` for the selected label.

## Inputs

- A `role="tablist"` of `role="tab"` buttons with `aria-selected`, roving tabindex and arrow keys, and the panel each tab controls.
- Optional: a shortcut key per tab.

## Rules

- `tri-tab` type (mono 0.62rem), padded `0.18rem 0.4rem`, 1px border of `gray` at 24%, square. Neighbours share one border.
- Rest label: `darkgray`. Selected: `tri-accent` fill and border, `on-accent` label. A best-efforts category may fill with its own category colour.
- A tab with a shortcut shows the key while unselected and its label when selected. The two spans swap width (`inline-size`, 180ms) and opacity (120ms); `data-motion="instant"` and reduced motion turn the transition off. The tab keeps one width per state because each span is sized in `ch`.
- Every tab, selected or not, shows the 2px `focus` ring outside the tab at 2px, over the ground (`focus.md`, Tabs). With a roving tabindex focus lands on the selected tab, so the ring must show over its fill.
- The row may scroll sideways on its own. Pad it 4px and pull it back with a -4px margin so the outside ring is not clipped (best-efforts categories pad 2px and inset the ring).
