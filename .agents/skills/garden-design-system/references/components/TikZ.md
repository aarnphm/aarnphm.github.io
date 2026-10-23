# TikZ

A TikZ diagram rendered at build time to inline SVG, with KaTeX labels and a source button.

**Source:** `quartz/plugins/transformers/tikz.ts` · styles: `quartz/styles/custom.scss` (`figure.tikz`), `quartz/components/styles/clipboard.scss` (`.clipboard-button`)

## Preview

A copy of the server-rendered markup from `/thoughts/MDS` (selector `figure.tikz`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/clipboard-pseudo.inline.ts`, with the shared `clipboard-button.ts`) on this markup.

## Rules

- Strokes inherit `dark`; raster fallbacks invert in the dark theme.
- The caption row holds an italic `source code` label and the copy button: a native `button.clipboard-button` of 24px with a 16px icon in `darkgray`. The icon swaps to a check once the source is copied.
