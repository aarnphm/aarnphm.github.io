# Instrument

A joined control: one `darkgray` frame around cells split by `line` hairlines, such as the PDF pager (`‹ | 4 / 36 | ›`) and zoom stepper (`− | 90% | + | fit`).

**Source:** mixins in `quartz/styles/pdf-instrument.scss` · inline embed: `quartz/styles/pdf-embed-toolbar.scss`, built by `createControls` in `quartz/components/scripts/pdf-common/viewer-chrome.ts` · `/read` reader: `quartz/components/styles/pdf-reader-controls.scss`, `quartz/components/pdf-reader/app.tsx`

## Preview

A copy of the inline PDF embed toolbar as the site renders it on `/thoughts/Navier-Stokes-equations` (selector `.pdf-embed-toolbar`, page 12 of 36 at 90%), styled by `site/garden.css`. The toolbar is built by script on the site, so the preview is static; the fit select and the loading status are left out.

## Inputs

- The cells in reading order: steppers either side of the value they change, a toggle that sets the same value last.
- The widest value each field can hold (the digit count of the page total, four characters for a zoom percent).
- An accessible name for the group (`Page`, `Zoom`) and for every glyph button.

## Rules

- The frame is 1px `darkgray` (4.97:1 on paper), square, `control-pdf` 1.5rem high, 44px under `(pointer: coarse)`. Cells split with a 1px `line` hairline on every cell but the first. `line` never draws the outer edge: at 1.43:1 it cannot mark where a control starts.
- Type is `code`, lowercase, `meta` with `tabular-nums`; 16px under a coarse pointer. A glyph cell is square with a 0.75rem icon; a word cell pads `0 0.5rem`.
- A value cell reserves its widest value and never wraps: the page field is `--pdf-instrument-digits` + 1 characters wide (written by `setPageCount`), the zoom readout four characters, so the instrument keeps one width while paging and zooming.
- Hover: `surface` fill and `dark` ink, only under `(hover: hover) and (pointer: fine)`; `:active` shows the same fill for touch. Pressed: `aria-pressed="true"` on `selected`. Focus: `2px solid dark` at `-2px`. Disabled: native `disabled` at opacity 0.5.
- Each instrument carries `role="group"` with an `aria-label`. A lone action beside the instruments (download) takes its own `darkgray` frame at the same height.
- Surfaces set the look through the `--pdf-instrument-*` custom properties (`height`, `font`, `icon`, `frame`, `line`, `surface`, `selected`, `digits`) from the `tokens` mixin; Arena's compact PDF controls override only those.
- Known debt: when `‹` or `›` becomes disabled while focused, focus drops to the body (native `disabled`). The `/read` reader's remaining buttons (margins, region, `?`, marks) still use the old 0.66rem frames.
