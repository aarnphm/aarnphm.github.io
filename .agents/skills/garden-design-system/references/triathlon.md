# Triathlon

The triathlon pages (`/triathlon`, day pages `/triathlon/on/YYYY/MM/DD`, sub-pages such as tools and the race and training calendars) form an analytics workspace inside the garden. They use the densest form of the tool register.

## Layout

- The page column is `width-tri-page`, `min(980px, 92vw)`. A timeline of days runs down it, separated by `tri-day-gap`.
- Day and activity cards expand in place. A full analysis opens in a modal of `width-tri-modal` (full screen on mobile).
- Components split by container queries, never by page breakpoints. Comparisons and wide plots get their own horizontal scroll.
- Analytics blocks open with an uppercase mono `tri-block-title`, followed by readouts (`Readout`), tabs (`TriTabs`) and one plot surface.

## Colour

- Sports: `tri-swim`, `tri-bike`, `tri-run` for fills and marks. Labels on paper use `calendar-swim`, `calendar-bike` and `calendar-run`, which are darkened in light and raw in dark.
- Provider and selection: `tri-accent` (Strava orange). Text on it is `on-accent`.
- Zones and physiology: `tri-zone-heart-rate`, `tri-zone-power`, `tri-zone-pace`, `tri-heart-rate`; the training effect in three steps; UV in five steps; the power profile (`tri-power-sprint`, `tri-power-attack`, `tri-power-climb`); and the model curve `tri-power-model`, drawn dashed because it is an estimate.
- State: `tri-health-*` for readiness, `tri-compare-better` and `tri-compare-worse` for deltas (always with a sign), and `exec-*` for training execution.
- Race series: `race-*` fills, each with a `-ink` for labels on paper and a fixed colour for text on the fill. Use them only to identify those organisations' series.

## Data rules

- Every value carries its unit, source and date. Native measurements, computed estimates and model outputs are labelled differently ("est.", a dashed line, a legend entry).
- Missing data stays missing ("–", a gap in the line). It is never drawn as zero.
- Numbers use `code` with tabular figures. Primary values use `tri-value`.
- A colour never carries meaning alone. Sport tags have words, execution has a status word, and deltas have signs.
- Charts take gridlines in `line`, axis labels in `darkgray` and values in `dark`. Hover readouts follow the pointer and are mirrored in a static table or caption.

## Interaction

- Keyboard hints (`KeyHint`) appear over controls in hints mode. Escape closes the analysis modal and returns focus to its trigger.
- Tabs are real tablists. Focus stays visible on the selected tab (`tri-be-focus`).
- The calendar's chosen view lives in the hash (`#training-<week>-list|week`). The width-based default applies only when no view has been chosen.

## Exemplars

- Copy `triathlon-best-efforts.scss` for a new analytics panel: an Arena-style framed list, tabular columns, and focus fixed locally.
- Copy the comparison workspace for multi-series plots on a single inspectable surface.
- Do not copy the `.tri-map-tab` selected pair (paper on `tri-accent`, 3.31:1) or the `gray` block titles (2.00:1).
