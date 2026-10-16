# CalloutCollapsed

A foldable callout in its collapsed state: only the title row and the fold control show.

**Source:** `quartz/plugins/transformers/ofm.ts` · styles: `quartz/styles/callouts.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/tsfm/4` (selector `blockquote.callout.is-collapsed`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/callout.inline.ts`, `quartz/components/scripts/collapse-header.inline.ts`) on this markup.

## Rules

- Written `> [!type]-` in Markdown. The fold control is a button with `aria-expanded`.
