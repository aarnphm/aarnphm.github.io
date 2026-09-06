## hover affordances without a script

a static (no-inline-script) figure can still get hover tooltips with `:has()` — useful for moving always-on legend prose into on-demand tips. wrap the hover target in a group with a **wide invisible hit-path** (so a thin arc is easy to hover), put the tooltip as an HTML overlay in a `position: relative` stage, and toggle from the figure root:

```scss
.x-stage {
  position: relative;
}
.x-arc-hit {
  fill: none;
  stroke: transparent;
  stroke-width: 14;
  pointer-events: stroke;
}
.x-tip {
  position: absolute;
  opacity: 0;
  pointer-events: none;
  transition: opacity 140ms ease; /* framed box */
}
.x-figure:has(.x-group--echo:hover) .x-tip--echo,
.x-figure:has(.x-legend-item--echo:hover) .x-tip--echo {
  opacity: 1;
}
```

mark up `<g class="x-group x-group--echo" tabindex={0} role="img" aria-label="…">` (the `aria-label` keeps the moved-to-tooltip prose accessible). RazorHeadTaxonomy does this for the echo/induction arc descriptions.

Provide the same explanatory content on keyboard focus. Tooltips must remain available when focus or hover moves to their content, and dismissible when they obscure other content. Keep essential information in an accessible description even if the visual tooltip is unavailable.
