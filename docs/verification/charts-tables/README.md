# charts-tables

## Change

`DataChart` (`quartz/components/mdx/DataChart.tsx`, `styles/dataChart.scss`, `scripts/data-chart.inline.ts`) is one chart component for bar and line charts: linear or log y, linear/log2/log10 x, up to two series, a listbox of x positions with a hover and keyboard readout, an optional ratio row, a native `details` table, and a probe mode that walks an analytic curve with a slider and a pointer drag. The component computes every state. The inline script only selects a state.

It replaces five chart images:

- `content/posts/structured decoding.md`: vLLM TTFT (log y) and TPOT. Values are as printed on the bars.
- `content/thoughts/sparse crosscoders.md`: loss against features and loss against FLOPs. Values were digitised from the Anthropic figure: the dot centroids were found by colour and the log y axis was fitted from its gridlines (`10.6` at y = 183.5, `5.8` at y = 488). The fit gives the other gridlines to within 0.3 px. Every dot sat about 3 px left of its gridline in the features panel, so the same correction was applied to the FLOPs panel. After that correction, the SAE points sit on integer powers of two and the crosscoder points sit at $k + 0.15$. Precision is about ±0.05 loss.
- `content/thoughts/economies of scale.md`: an illustrative LRAC curve, $36/Q + 2 + Q/4$, with sage marks at $Q = 4$ and $Q_2 = 12$, and a salmon probe.

Series colours: slot 0 is salmon, and slot 1 is sage mixed 70% into `--dark` (into `--lightgray` in the dark theme). Plain salmon and sage fail the dataviz validator's CVD check (deutan ΔE 2.5) and its normal-vision floor. The darker sage step passes both: protan ΔE 12.3 and normal ΔE 19.1 in light, and 10.1 and 17.0 in dark. The lightness-band and chroma checks fail for every Garden pastel, so each mark has a `--dark` outline and a table view, and each line series also has a marker shape.

## Replay

```
node docs/verification/charts-tables/replay.mjs
```

The replay takes the `DataChart` fences from the three notes as written and renders them through Quartz's `Codeblock` transformer (MDX parse, prop parsing, `htmlToJsx`). It uses `figure-harness.mjs` to open and check the page. It needs `node_modules` and Chromium at `/opt/pw-browsers/chromium`.

## Results

43/43 checks passed (`artifacts/results.json`):

- The tables show the printed values exactly.
- Each bar top is on its value, measured on a y scale fitted from the tick labels (error below 0.06 units). The bars share one baseline, keep a 2-unit gap and are at most 24 units wide.
- Each line marker is on a table value.
- Hovering each position shows that position's table row.
- Home, ArrowRight, End, the clamp at the ends and ArrowLeft work, and there is one tab stop. Focus shows a 2px ink stroke. Only the selected band is lit.
- The probe matches $36/Q + 2 + Q/4$ for each slider step from 8.5 to 14. Its region changes at the minimum, and Home and End work. The pointer drag works, the sage marks are on the curve, and `aria-valuetext` is correct.
- After a cleanup and a second `nav`, there are no duplicate listeners.
- The table disclosure opens.
- Light, dark and 320px screenshots exist, there is no horizontal page scroll and there are no page errors.

## Not checked

- The full Quartz build and the dev watcher (neither is available in this environment).
- Zoom at 200% and 400%.
- Touch dragging on a real device.
- Rendering of the Markdown tables that replaced the course-note images. The brief said that they need no harness run.
