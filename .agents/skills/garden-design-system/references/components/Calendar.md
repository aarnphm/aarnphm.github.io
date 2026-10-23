# Calendar

A month grid of days with event chips, a preview popover for each event, and a month jump.

**Source:** `quartz/components/controls/month-grid.tsx` (`MonthGrid`, the weeks of `section` days) and `controls/use-anchored-preview.ts` (`useAnchoredPreview`, the hover and focus preview), both Preact. The training calendar (`triathlon/calendar/TrainingMonth.tsx`) composes them; styles in `controls.scss` (`.g-calendar-*`), `triathlon-training-calendar.scss` and `triathlon-calendar.scss`.

**Skin:** the grid, day headers and nav buttons use the site's `controls.scss`. Today is the site's underline on the day number. The site has no chip or preview skin in `controls.scss` (triathlon styles its own by `data-sport`), so the `g-event` and `g-calendar-pop` rules in `bundle.css` belong to the system. A chip takes any tone (`swim`, `bike`, `run`, `salmon`, `iris`, `secondary`).

## Site API

- `MonthGrid({ id, dates, month, today, classes?, rowMinHeight?, weekdayLabel, dayLabel, dayAttributes?, dayHeader?, renderDay, ... })`. `dates` is a whole number of weeks; the host renders what each day holds. Default classes: `g-calendar`, `g-calendar-weekdays`, `g-calendar-grid`, `g-calendar-day`, `g-calendar-day-header`.
- `useAnchoredPreview({ canShow?, resetKeys, resetAnchor? })` keeps one `popover="manual"` preview under its anchor, placed with floating-ui. Escape or a click on an anchor dismisses it, and it stays dismissed until the pointer moves or focus moves to another anchor, so closing it cannot reopen it under a still pointer.

## Preview

The preview runs the site's Preact `MonthGrid` and `useAnchoredPreview` through `Garden.ui.calendar(root)`, in the same composition as `TrainingMonth.tsx`. The bundle carries Preact and floating-ui. Inputs:

- A root with `data-g-calendar`, `data-month` (`YYYY-MM`), and optional `data-id`, `data-today` and `data-noun` (the summary word, default "events").
- Events as JSON in a child `<script type="application/json">`: `{ id, date, title, tone, kind, minutes?, note? }`. `kind` goes into the chip label and the preview.

The binder renders the nav (previous, `today`, next), a month jump that is the site's date picker, the summary, the grid and the preview into a `.g-calendar-root` it appends, and returns a cleanup.

## Rules

- It renders 35 cells, or 42 when the month needs six weeks. Weeks start on Sunday.
- Each day is a `section` with an `h3 > time`. Today gets `aria-current="date"` and an underline. Days outside the month are `darkgray`.
- A chip is a button: a 2px tone mark, a one-line title with ellipsis, and mono meta. An empty day shows "–".
- The preview opens on pointer hover and on focus, sits under the chip (`bottom-start`, 6px gap) and flips when it would leave the viewport. It shows kind, date, duration and note. It closes 180 ms after the pointer leaves the chip and the preview, on blur, on a click on the chip, or on Escape (focus goes back to the chip). A dismissed preview stays closed until the pointer moves or focus moves to another chip.
- Changing the month closes the preview.
- The summary counts events and minutes in the shown month.
- In the triathlon calendar overlay and page, the panel bar (`tri-ana-bar`) holds the source switch (races or training) in the title slot and the close button at the right. Icon buttons in the bar show their localized `aria-label` as a tooltip that opens below, so the bar never clips it.
- Below 36rem of container width, chips drop the mark and the duration and show the title only.
