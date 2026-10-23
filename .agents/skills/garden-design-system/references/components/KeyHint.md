# KeyHint

A small mono chip above a control that shows its keyboard shortcut when hints are on.

**Source:** `quartz/components/styles/triathlon/timeline-preview.scss` (`.tri-key-anchor`, `.tri-key`, `.tri-hints .tri-key`) and `analytics-navigation.scss` (placement inside the analytics dialog).

## Inputs

- The shortcut text and the control it belongs to (`tri-key-anchor`).

## Rules

- `tri-key` type (mono 0.68rem), `darkgray` on paper, 1px `line` border, square, centred 0.35rem above its control.
- Hidden by default (opacity 0). Hints mode (`.tri-hints`) fades it in over 140ms and lifts it 0.25rem into place. It ignores the pointer and does not transition under reduced motion.
- Inside a dialog that clips above its top edge (the analytics tabs and the compare toggle) the chip drops below the control instead. On the search, close and calculator-close controls it sits inside the control, at the right or centred.
- The shortcut also appears in the control's accessible name or in a visible help list. The chip alone is not enough.
