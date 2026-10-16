# Byline

The block under the title: tags on the left and content metadata (dates, reading time) on the right.

**Source:** `quartz/components/Byline.tsx` · styles: `quartz/styles/custom.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `section.byline`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is static because its reading-time script (`content-meta.inline.ts`) fetches the page source.

## Rules

- Composes `TagList` and `ContentMeta` in a flex row that wraps on narrow screens.
- Dates and counts use `meta-mono`.
