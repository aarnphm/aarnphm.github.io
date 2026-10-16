# ExecutionLegend

The legend that names each training-execution colour and its range.

**Source:** `triathlon-training-calendar.scss` (`.tri-training-legend`).

## Inputs

- One item per execution state, with its range in words.

## Rules

- Swatch: 0.75rem square, the execution colour at 24% with a border in execution ink.
- Text: mono at `meta` size in `darkgray`, the range written beside each swatch.
- The list's `aria-label` is "execution colors"; each item's accessible name includes its range.
