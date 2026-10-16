# Callout

An Obsidian callout: a titled block for notes, examples and asides inside prose.

**Source:** `quartz/plugins/transformers/ofm.ts` · styles: `quartz/styles/callouts.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `blockquote.callout.note:not(.is-collapsed)`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/callout.inline.ts`) on this markup.

## Rules

- Types set a colour pair (`--color`, `--border`, `--bg`) and an icon.
- Math types (definition, theorem, proof and others) use MathCallout instead.
