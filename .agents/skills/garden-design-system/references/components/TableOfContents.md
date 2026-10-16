# TableOfContents

The minimal table of contents beside the note: a vertical rail of heading links that tracks the reading position.

**Source:** `quartz/components/TableOfContents.tsx` · styles: `quartz/components/styles/toc.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `#toc`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/toc.inline.ts`) on this markup.

## Rules

- Desktop only. Depth sets the indent; the active heading is filled at run time.
- Each entry is a link to its heading's anchor.
