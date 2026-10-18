# BaseToolbar

The Base view toolbar: the view menu, sort and filter controls, and search.

**Source:** `quartz/components/BaseViewSelectorMarkup.tsx`, `quartz/components/BaseSearchBar.tsx` (search and the filter and sort tabs) · styles: `quartz/components/styles/baseViewSelector.scss`, `quartz/components/styles/baseQuery.scss`

## Preview

A copy of the server-rendered markup from `/ideas/dynalist` (selector `.bases-toolbar`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/base-search.inline.ts`, `quartz/components/scripts/base-view-selector.inline.ts`) on this markup.

## Rules

- Buttons are `control-base` high. See Button and SearchField for the rules.
- `filter` and `sort` are a joined pair of `role="tab"` buttons in one 1px `line` frame (`.base-query-actions`). Only the tab whose panel is open takes the `selected` ground and `dark` text.
- A changed view has a modified marker. When the applied query differs from the `.base` file, the root carries `data-base-query-modified="true"` and the pair shows a 4px square in `dark` ink, 3px inside its top-right corner. The frame keeps the same `line` border on all four sides. Do not mark the state by recolouring one edge: in the dark theme `dark` is near-white, so a single bottom rule reads as a stray white line. `reset` in the query panel returns the view to the file's query and clears the marker. The panel shows "Unapplied changes." only for an edit not yet applied, which is a separate state.
