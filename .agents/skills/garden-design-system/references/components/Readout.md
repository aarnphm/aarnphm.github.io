# Readout

A bordered grid of labelled numeric readouts for data panels.

**Source:** the triathlon stat grids (`quartz/components/styles/triathlon/`, activity and day cards) and the data rules in `docs/design-patterns.md`.

## Inputs

- Per cell: label, value, unit, and whether the value is a native measurement, a computed estimate or a model output.

## Rules

- Label: `body` at `meta` size, `darkgray`. Value: `code` at 500, `dark`, tabular, `ui` size at most. Unit: `meta` size beside the value.
- A value cell reserves the widest value the data can produce, computed from the data (`ch` in the value font) with `white-space: nowrap`, so the grid keeps one layout across every state. The hairline figure readout (`figure-table` on a `dl`) follows the same rule at `meta`; see Compact tools.
- Mark an estimate with a word ("est."), not only a colour. Show a missing value as "–".
- Cells split by `line`. The grid wraps with `auto-fit`; the page never scrolls sideways.
- A sport colour appears as a small square swatch beside a written label.
