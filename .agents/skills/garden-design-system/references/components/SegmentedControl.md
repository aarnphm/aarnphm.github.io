# SegmentedControl

A joined row of square toggles where exactly one view is chosen.

**Source:** the segmented-choice pattern in `docs/design-patterns.md`, the triathlon calendar view controls (`triathlon-calendar.scss`, `.tri-calendar-view-controls`) and the Base view selector.

**Corrections:** the triathlon original is icon-only. This rendition adds text labels.

## Inputs

- Two to five native buttons with `aria-pressed`, or a tablist (`role="tab"`, roving tabindex, arrow keys) when the choice swaps a panel.

## Rules

- One outer `1px darkgray` border; segments share edges.
- Selected: `selected` fill, `dark` label, underline. The underline keeps the state visible without colour.
- Focus stays visible on the selected segment.
- Store the chosen view in the URL hash when it should survive a reload (`#training-<week>-list`).
