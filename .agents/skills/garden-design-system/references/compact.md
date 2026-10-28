# Compact tools

The working parts of the site (toolbars, readouts, small tables, figure chrome, Base views) are small and exact. Chrome has one text size, hairlines group, a frame marks where a control starts and stops, numbers never move, and nothing changes size with the window. This section gives the sizes and the four patterns that produce that look. The named source files are the truth; copy them before writing new CSS.

## Sizes

The root font size is 14px, so `1rem` is 14px. Sizes are rendered pixels.

| What                                                                                           | Size                                                                                                              | Token or rule                                                                                                         |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Prose                                                                                          | 16px                                                                                                              | `body`                                                                                                                |
| Control label, menu item, collection title, key value in a readout                             | 14px                                                                                                              | `ui` (`--text-ui`, 1rem); a key value adds weight 500                                                                 |
| Date, count, caption, tick, axis title, legend, readout row, table cell, toolbar label         | 12px                                                                                                              | `meta` (`--text-meta`, `6rem / 7`); numbers in `code` with `tabular-nums`                                             |
| Label inside a figure (node, lane, box, stage name)                                            | 12px, 13px at most                                                                                                | `meta`; 13px only where a one-line label must clear its box                                                           |
| Chart furniture outside the plot: legend, readout rows, the data table and its `table` summary | 10.5px                                                                                                            | `0.75rem` at the 14px root, `code` for numbers; the one size under `meta`, so the furniture sits under the 12px ticks |
| Superscript or exponent inside a figure                                                        | 9px                                                                                                               | `0.75em` of the 12px label                                                                                            |
| Control height                                                                                 | 1.5rem in readers and instruments, 1.75rem in collections, 2.125rem in Base views; 44px under `(pointer: coarse)` | `control-pdf`, `control-reader`, `control-collection`, `control-base`, `control-touch`                                |
| Text input on a phone                                                                          | 16px                                                                                                              | so iOS does not zoom the page                                                                                         |
| Hit band for a chart point                                                                     | at least 24px wide, the plot's full height                                                                        | a band narrower than 24px widens to 24px, centred on its point                                                        |
| Chart marker                                                                                   | 5.5px across (radius 2.75 at the rendered scale, plus a 1px stroke)                                               | `vector-effect: non-scaling-stroke`                                                                                   |

- Chrome has a floor and a ceiling: 12px to 14px, with one named exception at 10.5px (`0.75rem`): chart legends, readouts and data tables, and the PDF instrument labels. Labels at 0.62 to 0.66rem in Arena and the triathlon panels are debt, not precedent.
- A text size is set in px through the tokens. A viewBox unit, `vw`, or a container-query step that lands under 12px is not a text size.
- A size that differs between the inline frame, the Zoomable expanded frame and a 375px phone is a defect. Measure all three.

## Figure chrome holds its pixel size

Text inside an svg with a `viewBox` is sized in user units, so it grows and shrinks with the rendered width. On the site a 560-unit plot drawn at 496px turned 12px type into 9px, and the Zoomable frame moved every number again.

- `observeSvgScale(svgs, property)` in `quartz/components/scripts/figure-scale.ts` keeps `property` (for example `--x-u`) equal to user units per rendered px on each svg, with a `ResizeObserver`. It honours `preserveAspectRatio`: `meet` takes the larger axis ratio, `slice` the smaller. A figure without its own script sets `data-fig-scale="--x-u"` and `figure-scale.inline.ts` runs the observer for it.
- CSS multiplies: `font-size: calc(var(--x-u) * var(--text-meta))` for ticks, titles, legends, lane and node labels; `calc(var(--x-u) * 9px)` for a superscript. Markers scale the same way (`transform: scale(var(--x-u))` on a shape drawn at the origin of a translated group) and strokes use `vector-effect: non-scaling-stroke`, so a 1px line stays 1px.
- Each figure keeps a container-query fallback for `--x-u` (for example 0.85, 1, 1.15, 1.5 by frame width) for the paint before the script runs.
- When a phone frame would make the drawing smaller than its labels (labels hold 12px, boxes shrink under them), the drawing takes a minimum width (30 to 37rem in frames of 460 to 520px or less) and its wrapper scrolls sideways: a focusable `region` such as `.mha-scroll`, `.fat-scroll`, `.ag-scroll`. The page itself never scrolls sideways.
- Check: for every svg text and `foreignObject`, `getComputedStyle(el).fontSize` times the svg's px per unit reads 12.0 (13.0 for an in-figure label) at 1440 inline, in the expanded frame and at 375. `docs/verification/figure-text-scale/README.md` has the measurement helper and the per-figure numbers.

## Small tables and readouts

A readout under a figure, a stat strip, a properties list: all are hairline grids of square cells.

- Markup is a `<dl>` with `@include figure-table` (`quartz/components/styles/figure.scss`): columns `minmax(0, 1fr) max-content`, 1px gaps that show the `line` ground, cells padded `0.4rem 0.55rem`. `dt` is `fig-note` at weight 400; `dd` is `code`, `dark`, `tabular-nums`, right-aligned. Everything is 12px.
- A value column reserves the widest value the data can produce. Compute it where the data lives, at build time, through the same function that renders the state: `CrosscoderMasks.tsx` runs every variant and read layer through `stateFor` and takes the longest value in code points (`READOUT_VALUE_CH`, 14 today, from `L0 L1 L2 L3 L4`), then emits `--ccm-value-ch` on the `dl`. CSS reads it: `min-width: calc(var(--ccm-value-ch) * 1ch); white-space: nowrap`. `ch` resolves in the `dd` font, so the reservation is exact. A hand-picked `14ch` is not this rule: it drifts the day the data changes.
- The same rule sizes a toolbar field: the page field is one character wider than the digit count of the page total (`--pdf-instrument-digits`, written by `setPageCount`), and a zoom readout holds four characters, so `47%` and `152%` take the same width.
- Check by sweeping the states with a script and reading the container width: one number for every state.
- An estimate carries a word ("est."), a missing value is "–", a unit stays beside its number.

## The instrument

A pager (`‹ | 4 / 36 | ›`), a zoom stepper (`− | 90% | + | fit`) or any run of controls that act on one value is one instrument: one frame, cells inside it. The mixins live in `quartz/styles/pdf-instrument.scss` (`tokens`, `type`, `frame`, `cell`, `button`, `readout`, `zoom-readout`, `page-field`, `hover`, `focus`); the inline PDF embed (`quartz/styles/pdf-embed-toolbar.scss`) and the `/read` reader (`quartz/components/styles/pdf-reader-controls.scss`) include them under their own selectors. See the Instrument component.

- Frame: 1px `darkgray` (4.97:1 on paper), square. `line` reads 1.43:1 and fails the 3:1 a control boundary needs, so it draws only the hairlines between cells (`border-inline-start` on every cell but the first).
- Height: `control-pdf` 1.5rem; 44px under a coarse pointer. Every cell is `height: 100%` of the frame.
- Type: `code`, lowercase, `meta`, `tabular-nums`, `line-height: 1`; 16px under a coarse pointer so focusing the field does not zoom the page.
- Cells: a glyph cell is square (`aspect-ratio: 1`, icon 0.75rem); a word cell pads `0 0.5rem`; a value cell is `white-space: nowrap` with a reserved width (see above). The group carries `role="group"` and an `aria-label`.
- States: hover is `surface` fill with `dark` ink under `(hover: hover) and (pointer: fine)`, and `:active` shows the same fill for touch; a pressed toggle (`fit`) is `aria-pressed="true"` on `selected`; focus is `2px solid dark` at `-2px`; disabled is native `disabled` at opacity 0.5. Transitions are 120ms on `background-color` and `color`.
- Order: the value sits between its two steppers; a toggle that sets the same value (`fit`) is the last cell of that stepper. A lone action (download) carries its own `darkgray` frame at the same height, so the row reads as one family.
- The Base `filter` / `sort` pair (BaseToolbar) is the same idea with a `line` frame. A new joined control takes the `darkgray` frame.

## A Base-like page

A page that lists, filters and reads many items (a Base view, a tag or folder list, stream, an Arena channel) composes the patterns above. See the Tool surfaces section for each page's density and the BaseToolbar, CollectionRow and Table components.

- One toolbar in one stable place: the view name, the `filter` / `sort` instrument, search. Controls are `control-base` 2.125rem (1.75rem in collections) in `ui` at weight 500, lowercase, named by what they do.
- Below it a hairline list or table on `line`, rows that are links, `surface` on hover, `selected` on a match. Row metadata (date, count, property) is `meta`; numbers are `code` with `tabular-nums`.
- No cards, no shadows, no rounded corners, no decorative left border. A frame that marks a boundary is `dark` or `darkgray`; everything else groups by hairline, fill or space.
- The table scrolls inside its own container; the page body never scrolls sideways down to 320px. Grid and flex children that hold text get `min-width: 0`.
- At phone width the toolbar wraps; search keeps a full row, and the controls stay 44px.

## Before calling it done

1. Measure every chrome text size by script in the three frames; none under 12px, none over 14px, in-figure labels 12 to 13px.
2. Sweep the states of every changing value and record the container width; it must not change.
3. `document.documentElement.scrollWidth` equals `clientWidth` at 375; only figure scrollers scroll.
4. Frames reach 3:1 and text 4.5:1 in both themes; hit targets reach 24px, 44px under coarse pointers.
5. Save the route, viewport, theme and screenshots under `docs/verification/<task>/artifacts/` with the measurement script.
