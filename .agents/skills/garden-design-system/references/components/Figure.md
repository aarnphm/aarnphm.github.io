# Figure

A square-framed technical figure with KaTeX labels and a caption, used inside notes.

**Source:** `quartz/components/styles/figure.scss` (`figure-palette`, `figure-frame`, `figure-fo`, `figure-caption`).

**Corrections:** the source caption mixes `gray` 70% with `dark` (3.68:1 on paper). This rendition uses `fig-caption` (`darkgray`, 4.97:1).

## Inputs

- The figure body: SVG, controls or readouts.
- Symbol labels as KaTeX inside `foreignObject` elements.
- A caption that says what the figure shows and what to look at.
- A container name for the figure's own container queries.

## Rules

- Frame: `1px solid fig-frame`, square, padded `space-6 1.75rem`, on paper.
- Accents: `salmon` and `secondary`. Strokes: `darkgray`. Dividers: `line`.
- Labels: `math-figure` (0.95em), colour inherited from the label.
- Caption: 0.78rem / 1.45 in `fig-caption`, below the body.
- Large or interactive figures go inside the site's `Zoomable` wrapper.
