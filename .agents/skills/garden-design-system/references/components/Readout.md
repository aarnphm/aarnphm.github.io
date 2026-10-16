# Readout

A bordered grid of labelled numeric readouts for data panels.

**Source:** the triathlon stat grids (`triathlon.scss`, activity and day cards) and the data rules in `docs/design-patterns.md`.

## Inputs

- Per cell: label, value, unit, and whether the value is a native measurement, a computed estimate or a model output.

## Rules

- Label: `body` at `meta` size, `darkgray`. Value: `code` at 500, `dark`, tabular. Unit: `meta` size beside the value.
- Mark an estimate with a word ("est."), not only a colour. Show a missing value as "–".
- Cells split by `line`. The grid wraps with `auto-fit`; the page never scrolls sideways.
- A sport colour appears as a small square swatch beside a written label.
