# Breadcrumbs

The path from `~` to the current note, separated by `/`, with middle segments folded into an overflow trigger.

**Source:** `quartz/components/Breadcrumbs.tsx` · styles: `quartz/components/styles/breadcrumbs.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `nav.breadcrumb-container`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/breadcrumbs.inline.ts`) on this markup.

## Rules

- Root name `~`, separator `/`; three leading and two trailing segments on notes, one trailing segment on list pages.
- Segments are links in `darkgray`; the current page is plain text in `dark`.
