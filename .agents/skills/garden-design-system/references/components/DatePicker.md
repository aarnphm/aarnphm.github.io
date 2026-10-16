# DatePicker

A date field that opens a month grid in the top layer.

**Source:** `quartz/components/controls/date-picker.ts` (`buildDatePicker`), with date helpers in `controls/dates.ts` and the open/close model in `controls/date-picker-model.ts`. Skin: `quartz/components/styles/controls.scss`. Used by the pace forecast, the distributions panel, the training calendar's month jump and other triathlon modules.

**Skin:** the preview loads the site's `controls.scss` as shipped. The selected day has a `secondary` tint (24% over the control surface) and `dark` text. Today is the day number underlined at weight 600, in text ink, the same mark the month grid puts on its day header; it stays when today is also selected. The focused day has the `focus` ring inset in the cell.

## Site API

- `buildDatePicker({ id, label, labels?, selected, min?, max?, onOpen?, onSelect, onClear? })` returns `{ wrap, trigger, text, panel, render, close, mount }`. `selected`, `min` and `max` are functions, so the picker reads the host's current state each time it opens. The host writes the formatted value into `text`.
- The trigger (`.g-datepicker-trigger`, `aria-haspopup="dialog"`) holds `.g-datepicker-text` and a `.g-datepicker-icon`. The panel is `role="dialog"` with `popover="auto"`, named "label · date picker".
- Days are `.g-cal-day` buttons named by their ISO date. The selected day has `aria-pressed="true"`; today has `aria-current="date"`.
- `labels` localises the strings, month titles and weekday letters; `englishDateLabels` is the default. Firing `datepicker:render` on the panel redraws an open picker.

## Preview

The preview runs the site's `buildDatePicker` through `Garden.ui.datePicker(root, { onSelect?, format? })`. The root is a `div` with `data-g-datepicker`, `data-label`, and optional `data-id`, `data-value`, `data-min`, `data-max` and `data-placeholder` (default "choose a date"). The binder reads `data-value`, `data-min` and `data-max` each time the panel opens, writes the choice back to `data-value`, puts `format(value)` in the trigger text, dispatches `g:change`, and returns a cleanup. `clear` removes `data-value` and dispatches `g:change` with `null`.

## Rules

- Dates are local calendar days as ISO strings. Weeks start on Sunday. The grid is always 6 × 7, so the panel does not change height between months.
- Arrow keys move by a day or a week, PageUp and PageDown by a month, Home and End by six weeks. The value stays inside `min` and `max`; days outside are disabled.
- The foot has `clear` and `today`. Choosing a day closes the panel and returns focus to the trigger.
- The panel is placed under the trigger and flips above it when it would leave the viewport. A click outside closes it.
