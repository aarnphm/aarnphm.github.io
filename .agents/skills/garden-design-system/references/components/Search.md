# Search

The search button and its open panel: a text field, a lexical or semantic mode toggle and a result list.

**Source:** `quartz/components/Search.tsx` · styles: `quartz/components/styles/search.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `.search:not(.bases-search)`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is static because it fetches the site's content index.

## Rules

- Opens from the button or the keyboard shortcut; Escape closes it.
- The mode toggle is a radio group of buttons with `aria-pressed`.
- The preview shows the panel open with no query; results render from the search index at run time.
