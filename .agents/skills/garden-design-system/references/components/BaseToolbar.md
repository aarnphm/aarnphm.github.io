# BaseToolbar

The Base view toolbar: the view menu, sort and filter controls, and search.

**Source:** `quartz/components/BaseViewSelectorMarkup.tsx` · styles: `quartz/components/styles/baseViewSelector.scss`

## Preview

A copy of the server-rendered markup from `/ideas/dynalist` (selector `.bases-toolbar`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/base-search.inline.ts`, `quartz/components/scripts/base-view-selector.inline.ts`) on this markup.

## Rules

- Buttons are `control-base` high. See Button and SearchField for the rules.
