# AttributionGraph

The acronym attribution graph of Circuit Tracing (figure 5): supernodes from the prompt tokens to the DAG logit, with ablation and edge pruning.

**Source:** `quartz/components/mdx/AttributionGraph.tsx` · model: `quartz/util/attribution-graph.ts` · styles: `quartz/components/styles/attributionGraph.scss` · script: `quartz/components/scripts/attribution-graph.inline.ts`

## Preview

A copy of the server-rendered markup from `/thoughts/mechanistic-interpretability` (selector `.attribution-graph`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/attribution-graph.inline.ts`) on this markup.

## Rules

- Wrapped in `Zoomable` on the site. Follows the figure grammar: `fig-frame` with a 34rem intrinsic height, `salmon` for the selected node, `secondary` (sage) for the activation bar under each feature, `fig-stroke` edges and `fig-note` axis text.
- Every node is a `<g role="button" tabindex="0">`. Hover or focus selects it and fills the readout with its inputs. Click, Enter or Space ablates it (`aria-pressed`), and the replacement model recomputes every node above it. The logit is `aria-disabled` and stays. "restore ablated nodes" is `hidden` until something is ablated.
- The prune control is the KVHeadGrouping control grid (`max-content minmax(0, 1fr)`): label in `fig-note` at 0.82rem, slider on `stableRangeControl` from `quartz/styles/mixin.scss` (0.2rem track, 1rem square thumb with a `gray` border). Its `<output>` is `code` at `meta` size with tabular figures: threshold, edges kept, share of attribution kept.
- A pruned or zero edge stays in place at `fig-stroke-soft` and 55% opacity, so the graph keeps its shape. Edges into the selected node draw in `dark`. Stroke width is 0.8 + 2.4|A| viewBox pixels; arrowheads are `fig-stroke`.
- The readout is the `figure-table` hairline `dl` (`aria-live="polite"`): node, activation (written `a₀ → a` once an ablation changed it), a `selected`-filled head row, then one row per input edge plus the error node or "other paths".
- The drawing keeps a 36rem minimum width and scrolls inside `.ag-scroll`, a focusable region; the stage goes to two columns (drawing, readout) above a 44rem container. The page never scrolls sideways.
- The weights are illustrative, chosen so each node's inputs sum to one; the caption says so. Transitions are 140 to 180ms on named properties and stop under reduced motion.
