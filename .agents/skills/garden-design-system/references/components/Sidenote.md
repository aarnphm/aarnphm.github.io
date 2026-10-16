# Sidenote

A sidenote: a numbered label in the text whose note sits in the margin on wide screens and expands inline on narrow ones.

**Source:** `quartz/plugins/transformers/sidenotes.ts` · styles: `quartz/components/styles/sidenotes.inline.scss`

## Preview

A copy of the server-rendered markup from `/stream/on/2025/10/30` (selector `p:has(.sidenote)`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/sidenotes.inline.ts`) on this markup.

## Rules

- The label is a control with `aria-controls`; the note content is placed at run time.
