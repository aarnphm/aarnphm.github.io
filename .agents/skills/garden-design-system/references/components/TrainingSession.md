# TrainingSession

A training-calendar card for one planned or completed workout, tinted by how closely it matched the plan.

**Source:** `triathlon-training-calendar.scss` (`.tri-training-session`, `.tri-training-workout-kind`, `.tri-training-sport`, `.tri-training-status`, `.tri-training-workout-title`).

## Inputs

- Sport, execution state (`on-target`, `off-target`, `far-off-target`, `missed`, `planned`), title, planned and actual metrics.
- A link to the matched activity, when one exists.

## Rules

- Ground: the execution colour at 16% over paper (24% when selected). A `calendar-line` bottom rule; no frame, square.
- Kind row (`tri-kind`): a sport tag on the left (sport colour at 32% over paper, ink label) and the status word on the right in execution ink (execution colour 65% + `dark`).
- Title: `tri-session-title`, one line with ellipsis. Metrics: mono at `meta` size, tabular.
- The colour always comes with the status word. A missing value shows "–".
- Execution bands: on target 80 to 120% of plan; off target 50 to <80% or >120 to 150%; far off outside 50 to 150%; missed for a past session with no recording; neutral for a plan.
