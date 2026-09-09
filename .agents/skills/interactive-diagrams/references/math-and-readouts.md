# Math, readouts, and sliders

Use KaTeX for mathematical labels and numeric readouts, including a slider's value chip. Keep request text and code-like labels as ordinary text. Format units and symbols as TeX (`\\%`, `\\times`, `\\infty` in TypeScript strings) and reuse the component's render helper.

For server rendering, use the existing `katexOptions` and `customMacros` from `quartz/cfg.ts`. Browser helpers should import `katex` directly and include only the macros they need; the server configuration is not a browser dependency. Keep server and browser formatting consistent. Never return unescaped source text from a catch block into an `innerHTML` sink.

Choose the update path from the value set:

- For a small finite set, pre-render values in TSX and toggle visibility. `KVHeadGrouping.tsx` and `kv-head-grouping.inline.ts` use this approach.
- For continuous values, update through a local `katex.renderToString` helper with `output: 'html'` and `throwOnError: false`. `cascade-filter.inline.ts` contains the existing pattern. Recompute only when its inputs change.

Use `<dl>` for label/value statistics, the shared frame border for readout cards, and `figure-card-heading` for their headings. Keep readout width stable with tabular numerals and appropriate minimum width.

For native range inputs, reuse `stableRangeControl` from `quartz/styles/mixin.scss`. Update the input value, displayed readout, and component-specific CSS fill variable together. The native input exposes its numeric value; set `aria-valuetext` when a formatted unit or interpretation adds meaning. Custom sliders also need their full keyboard and ARIA contract.

## Clipping and alignment

- SVG `<text>` cannot render KaTeX. Use `foreignObject` and `figure-fo`; set `justify-content` at the call site for the intended horizontal alignment.
- `foreignObject` can clip even when its inner div has `overflow: visible`. Size the element generously and inspect its own overflow.
- The frame's `content-visibility: auto` can clip oversized display math. Give the inner display-math block horizontal overflow when necessary and verify it in the rendered figure.
- For colliding adjacent labels, adjust their anchors and available space before shrinking text. Keep stacked non-math labels as text so glyph bearings do not control alignment.
