# Math and numbers

The garden renders LaTeX with KaTeX at build time (`Plugin.Latex`, strict mode, `throwOnError`). 280 notes contain display math and 17 contain TikZ diagrams, so math is part of the reading register. It sets in `KaTeX_Main` beside PP Neue Montreal, in `math-ink`.

## Where math goes

| Context       | Style                  | Rule                                                                                                                             |
| ------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Prose         | `math` (1.1em)         | Inline `$…$` in a sentence. Display `$$…$$` on its own lines, centred, `space-3` above and below.                                |
| Wide display  | `math`                 | Scrolls sideways inside `.katex-display`. The page never scrolls.                                                                |
| Headings      | `math-heading` (1em)   | 0.2em side margins.                                                                                                              |
| Math callouts | `math`                 | 1px `math-frame`, no fill, square. Display math gets `space-4` above and below. See MathCallout.                                 |
| Figures       | `math-figure` (0.95em) | Symbol labels sit in a `foreignObject` (`figure-fo`). Caption math inherits the caption colour.                                  |
| TikZ          | `math-figure`          | A ` ```tikz ` fence renders to inline SVG with KaTeX labels. Raster fallbacks invert in dark: `invert(0.88) hue-rotate(180deg)`. |
| Pseudocode    | `pseudocode`           | Keywords at 700, procedure names in small caps, line numbers at 0.8em.                                                           |

## Numbers

A number in an argument sets in math, in the same face as the symbols around it. A number that a tool updates, or that a reader scans down a column, sets in `code` with tabular figures so the digits align.

|                      | Prose and figures                        | Tools and data              |
| -------------------- | ---------------------------------------- | --------------------------- |
| Face                 | inline math                              | `meta-mono`, `tabular-nums` |
| Thousands            | `500{,}000`                              | `500,000`                   |
| Units                | `28\,\mathrm{mm}`, `3.35\,\mathrm{TB/s}` | `28 mm`                     |
| Multiples and shares | `2.3\times`, `80\%`                      | `2.3×`, `80%`               |
| Powers               | `10^{-8}`, `2^{64} - 1`                  | `1e-8` in code only         |
| Time, pace, dates    | rare                                     | `4:27/km`, `2026-10-07`     |

- Units are upright (`\mathrm{}`) and follow a thin space (`\,`). Notes mix `\mathrm{}`, `\text{}` and `\ ` today. New writing uses `\,\mathrm{unit}`.
- Named quantities use `\text{FLOPs}` or `\operatorname{softmax}`. `quartz/cfg.ts` defines `\argmin`, `\argmax` and `\abs{x}`.
- Inside a Markdown table, write absolute values as `\lvert x \rvert`. A literal `|` splits the cell.
- KaTeX runs strict, so an unsupported command fails the build. `\multicolumn` is a shim that keeps only the cell content.

## Charts

- Tick labels: `meta-mono`, `darkgray`, tabular. Log axes use SI suffixes (`1k`, `10k`).
- Axis titles: words in `body` at `meta` size, unit and scale in parentheses: "arithmetic intensity (FLOP/byte, log)".
- Symbols on the plot (`\beta`, `C`, a ridge point) are KaTeX labels in a `foreignObject`, never SVG text.
- Gridlines in `line`; data and model strokes in `dark`; reference lines in `darkgray`, dashed. A computed estimate is dashed too.
- Salmon marks the one element to look at. Structure stays in `dark`, `darkgray` and `line`, and a second series is `secondary`. Each series also gets a second cue: marker shape, dash pattern or a direct label.
- The caption names the source or hardware, gives the key number with its unit, and says what the reader should see.

## Colour

`math-ink` is `dark` in both themes. `base.scss` sets `color` and `fill` on `.katex`, `.math` and typst output. Figure labels inherit the figure's ink. Math callouts draw their frame in `math-frame` (`dark`). Proof reasons use `darkgray`; the source uses `gray`, which reads 2.00:1.

## Debt

- The site loads KaTeX CSS 0.16.11 from jsdelivr but renders with the installed `katex` 0.17.0. Pin both to the same version.
- `.katex` sets its font with `!important`.
- On mobile, TikZ figures shrink with `transform: scale(0.7)`, which leaves the layout box at full size.
- Figure captions once mixed `gray` 70% with `dark` (3.68:1). The source now sets them in `fig-caption` (`darkgray`) at the `meta` size, as this system does.
