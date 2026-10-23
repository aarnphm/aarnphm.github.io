# Chart

A data chart inside a figure frame: mono ticks, worded axis titles, KaTeX annotations and a caption with the key number.

**Source:** the figure grammar in `quartz/components/styles/figure.scss` and the chart rules in `docs/design-patterns.md`. The data is a roofline for one H100 SXM in BF16 (C = 989 TFLOP/s, β = 3.35 TB/s).

## Inputs

- The data, its units, scale (linear or log) and source.
- Annotations for named quantities.
- A caption that states the source, the key number and what to see.

## Rules

- Ticks: `meta-mono`, `darkgray`, tabular. Log axes use SI suffixes (`1k`, `10k`).
- Axis titles: words at `meta` size in `darkgray`, with unit and scale in parentheses.
- Gridlines `line`. Data and roof strokes `dark`. Reference lines `darkgray`, dashed.
- Symbols such as `\beta`, `C` and the ridge point are KaTeX labels in `foreignObject` elements.
- Salmon marks the one element to look at (here the decode point, which the caption names). Data and roof strokes stay `dark`, reference lines `darkgray` and gridlines `line`. A second series is `secondary`.
- Each series also has a second cue: here a salmon circle and a sage square, each with a direct label.
- A computed estimate draws dashed. Missing data leaves a gap.
