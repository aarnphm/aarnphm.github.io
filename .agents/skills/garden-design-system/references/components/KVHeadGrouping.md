# KVHeadGrouping

Grouped-query attention: query heads sharing KV heads, with a cache-bandwidth readout.

**Source:** `quartz/components/mdx/KVHeadGrouping.tsx` · styles: `quartz/components/styles/kVHeadGrouping.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/GQA` (selector `.kv-head-grouping`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/kv-head-grouping.inline.ts`) on this markup.

## Rules

- Wrapped in `Zoomable` on the site. Interaction (sliders, steps, hover readouts) runs from the component's inline script; the preview shows its first rendered state.
- Follows the figure grammar: `fig-frame`, `salmon` and `secondary` accents, KaTeX labels in `foreignObject`.
