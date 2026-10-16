# Palette

Figures use the Garden theme (`quartz.config.ts` `theme.colors`) through the `--fig-*` tokens that `figure-palette` sets on every figure root. Resolved values and contrast ratios are in the `garden-design-system` skill, `references/tokens.md`.

## Tone ladder

The `anatomy` skill shades a figure on a ladder of `faint`, `lo`, `mid`, `hi` and one `lit` part. Garden figures use the same ladder with these tokens:

| Step | Token | Use |
| --- | --- | --- |
| ground | `--light`, `--fig-surface` | Figure background; fills that hide what is behind them |
| faint | `--fig-div`, `--fig-line` | Gridlines, cell borders, dividers inside cards |
| lo | `--fig-stroke-soft` | Quiet structure: inactive edges, empty slots |
| mid | `--fig-stroke` | Part outlines and active edges |
| hi | `--dark` | The subject's strokes, labels, math, values |
| lit | `--fig-salmon` fill, `--dark` stroke | The one part the reader watches now |
| second state | `--fig-sage` fill | Done, kept, shared, cached |
| frame | `--fig-frame-border` | The 1px figure frame and readout cards |

Text sits on its own scale: `--dark` for labels and values, `--fig-note` for card headings, `--fig-caption-fg` (`--darkgray`) for captions, `--fig-tab-fg` for tab labels.

## Rules

- **Accents are fills.** Salmon reads 1.69:1 and sage 1.51:1 on light paper, so neither carries text or a thin meaningful line there. A salmon node keeps a `--dark` or `--fig-stroke` outline; a label on a salmon fill is `--dark`.
- **Define component tokens from figure tokens.** Write `--kvg-salmon: var(--fig-salmon)` and `color-mix(in srgb, var(--fig-salmon) 30%, var(--light))`, never the hex. The figure root includes `figure-frame`, so `--fig-*` resolve on the same element.
- **Dark fills mix into `--lightgray`.** In the dark theme, a tint over `--light` disappears; the existing figures mix 18 to 26% accent into `--lightgray` under `:root[saved-theme='dark'] .figure-root`. Keep accent strokes at full value.
- **Lightness tells states apart.** Two states that must be distinguished differ in lightness or fill weight as well as hue, and the legend or caption names them.
- **A third series** uses `--tertiary` (peach) for hints or Flexoki iris (`#8b7ec8`, via a figure token) for a tag, with a reason the caption can state. More than three colours means the figure is showing several mechanisms; split it.
- **Name colours by the palette.** Captions and legends say salmon and sage.
- **No hex in component SCSS.** `figure.scss` holds the only literal (`--fig-salmon`, which has no theme variable). `--fig-sage` reads `--secondary`.
