# Byline

The block under the title: tags and media on the left and content metadata (dates, reading time, share links, source) on the right.

**Source:** `quartz/components/Byline.tsx` · styles: `quartz/styles/custom/article-chrome.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `section.byline`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static because its reading-time script (`content-meta.inline.ts`) fetches the page source.

## Rules

- Two columns that wrap on narrow screens. Left: `étiquette` (tags) and `média`. Right: `publié le`, `modifié le`, `durée`, `partager` (`carte`, `citation`, `liens`) and `source` (`llms.txt`).
- Labels are small uppercase text in `darkgray`; dates and counts use `meta-mono`.
