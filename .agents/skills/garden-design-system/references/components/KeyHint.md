# KeyHint

A small mono chip above a control that shows its keyboard shortcut when hints are on.

**Source:** `triathlon.scss` (`.tri-key`, `.tri-hints .tri-key`).

## Inputs

- The shortcut text and the control it belongs to.

## Rules

- `tri-key` type (mono 0.68rem), `darkgray` on paper, 1px `line` border, square, centred 0.35rem above its control.
- Hidden by default (opacity 0). Hints mode fades it in over 140ms. It ignores the pointer.
- The shortcut also appears in the control's accessible name or in a visible help list. The chip alone is not enough.
