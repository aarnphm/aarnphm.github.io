# MetaEthicsTree

A decision tree of meta-ethical positions, starting from whether moral claims can be true or false.

**Source:** `quartz/components/mdx/MetaEthicsTree.tsx` · styles: `quartz/components/styles/metaEthicsTree.scss`

## Preview

A copy of the server-rendered markup from `/personality/slides` (selector `.meta-ethics-tree`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/meta-ethics-tree.inline.ts`, `quartz/components/scripts/slides.inline.ts`) on this markup.

## Rules

- Wrapped in `Zoomable` on the site. Interaction (sliders, steps, hover readouts) runs from the component's inline script; the preview shows its first rendered state.
- Follows the figure grammar: `fig-frame`, `salmon` and `secondary` accents, KaTeX labels in `foreignObject`.
