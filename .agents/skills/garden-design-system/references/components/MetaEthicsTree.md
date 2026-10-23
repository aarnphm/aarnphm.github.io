# MetaEthicsTree

A decision tree of meta-ethical positions, starting from whether moral claims can be true or false.

**Source:** `quartz/components/mdx/MetaEthicsTree.tsx` · styles: `quartz/components/styles/metaEthicsTree.scss`

## Preview

A copy of the figure from `/personality`, where the note embeds it (selector `.meta-ethics-tree`, with the default answer, moral objectivism, selected), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/meta-ethics-tree.inline.ts`) on this markup.

## Rules

- Wrapped in `Zoomable` on the site. Interaction (sliders, steps, hover readouts) runs from the component's inline script; the preview shows its first rendered state.
- Follows the figure grammar: `fig-frame`, KaTeX labels in `foreignObject`. Salmon marks only the selected path and its answer; the other terminal positions are `secondary` and the questions are ink on paper. Selecting a position moves the salmon path and fills the readout beside the tree.
