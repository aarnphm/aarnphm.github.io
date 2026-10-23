# Figure

A square-framed technical figure with KaTeX labels and a caption, used inside notes.

**Source:** `quartz/components/styles/figure.scss` (`figure-palette`, `figure-frame`, `figure-fo`, `figure-caption`, `figure-grid`, `figure-fieldset`, `figure-table`) · type scale: `quartz/components/scripts/figure-scale.ts`

## Inputs

- The figure body: SVG, controls or readouts.
- Symbol labels as KaTeX inside `foreignObject` elements.
- A caption that says what the figure shows and what to look at.
- A container name for the figure's own container queries.

## Rules

- Frame: `1px solid fig-frame`, square, padded `1.5rem 1.75rem`, on paper, with `1.75rem` above and below. It is a size container, so a figure answers the width of its own column.
- Salmon marks the one element to look at. Structure is `dark`, `darkgray` and `line`. A second series or state is `secondary`. Accents are fills only (salmon is 1.69:1 and sage 1.51:1 on paper), so a label on one is `dark` and a salmon node keeps a `dark` or `darkgray` outline. A third colour needs a reason the caption can name.
- Strokes: `darkgray`. Dividers: `line`. Corners are square and the stroke is solid.
- Labels: `math-figure` (0.95em), colour inherited from the label.
- Caption: `meta` size at 1.45 in `fig-caption`, below the body. Colour words in it name the palette (salmon, sage) and match what the reader sees.
- Type inside a scaled SVG holds its pixel size: a resize observer keeps `user units per rendered px` on each svg, and a label's size is that factor times a pixel value, so text does not shrink with the frame.
- Readouts below the figure use the hairline grid (`figure-grid`, `figure-table`): the ground shows through 1px gaps, so every cell edge is one line. A control group is a `fieldset` whose `legend` sits clear of the grid (`figure-fieldset`).
- Large or interactive figures go inside the site's `Zoomable` wrapper.
