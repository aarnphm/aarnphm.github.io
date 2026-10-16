# CodeBlock

A code block from rehype-pretty-code: an optional title, line numbers and syntax colours in both themes.

**Source:** `quartz/plugins/transformers/syntax.ts` · styles: `quartz/styles/syntax.scss`

## Preview

A copy of the server-rendered markup from `/puzzle/lc` (selector `figure[data-rehype-pretty-code-figure]`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/clipboard.inline.ts`, `quartz/components/scripts/code-copy.inline.ts`) on this markup.

## Rules

- Set in `code` type. The copy button is injected at run time by CodeCopy.
- Wide code scrolls inside the block.
