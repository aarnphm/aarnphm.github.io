# Math

Inline and display math in prose, and the two ways to write a number.

**Source:** `quartz/styles/custom.scss` (`.katex`, the 1.1em face), `quartz/styles/custom/article-chrome.scss` and `quartz/styles/custom/callouts.scss` (`.katex` in articles and callouts), `quartz/styles/base/document-defaults.scss` (`.katex`, `.katex-display`, `.math`), `quartz/styles/base.scss` (`.katex-html` and inline overflow), `quartz.config.ts` (`Plugin.Latex`), `quartz/cfg.ts` (macros, strict mode).

## Inputs

- LaTeX in `$…$` or `$$…$$`. The build renders it with KaTeX; nothing renders in the browser.

## Rules

- `math` style: `KaTeX_Main` at 1.1em of the surrounding text, `math-ink`.
- Inline math that is wider than the column scrolls inside itself (`:is(p, li) > .katex` is `inline-flex` with `overflow-x: auto`), and the line keeps its baseline.
- Display math centres and sits `space-3` above and below. When it is wider than the column, it scrolls inside `.katex-display`.
- A number in an argument is inline math: `500{,}000`, `28\,\mathrm{mm}`, `2.3\times`, `10^{-8}`.
- A number in a tool, table or readout is `meta-mono` with tabular figures: `500,000`, `28 mm`, `4:27/km`.
- Units are upright after a thin space: `\,\mathrm{TB/s}`.
