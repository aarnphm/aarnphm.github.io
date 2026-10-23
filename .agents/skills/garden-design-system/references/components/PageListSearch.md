# PageListSearch

The filter field above a folder or tag list.

**Source:** `quartz/components/PageListSearch.tsx` · styles: `quartz/components/styles/listPage.scss`

## Preview

A copy of the server-rendered markup from `/tags/drama` (selector `.page-listing`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/pagelist-search.inline.ts`) on this markup.

## Rules

- Focus: the input removes its own outline and the field frame takes the 2px `focus` ring through `:has(> input:focus-visible)` (see `focus.md`, item 3).
- Filters rows in place and keeps the list frame visible while any row matches.
