# Transclude

A transcluded block: another note's section embedded in place, with a link back to its source.

**Source:** `quartz/components/renderPage.tsx` · styles: `quartz/styles/base.scss`

## Preview

A copy of the server-rendered markup from `/stream/on/2025/11/28` (selector `blockquote.transclude`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is static because it fetches the linked page.

## Rules

- Written `![[note#heading]]`. The embed renders at build time.
