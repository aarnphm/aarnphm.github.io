# Evergreen

The evergreen aside on a folder page: the folder's most-linked notes with a count.

**Source:** `quartz/components/Evergreen.tsx` · styles: `quartz/components/styles/evergreen.scss`

## Preview

A copy of the server-rendered markup from `/thoughts` (selector `aside.notes-evergreen`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/evergreen.inline.ts`) on this markup.

## Rules

- The count uses `meta-mono`.
