# Keybind

The keyboard-shortcut help: the shortcut chip and the list of bindings it opens.

**Source:** `quartz/components/Keybind.tsx` · styles: `quartz/components/styles/keybind.scss`

## Preview

A copy of the server-rendered markup from `/` (selector `.keybind`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#` and lists keep their first 8 items. The preview is interactive: it calls `Garden.mount()` from `components/bundle.js`, which runs the site's own script (`quartz/components/scripts/keybind.inline.ts`) on this markup.

## Rules

- Bindings are written as `kbd` chips in mono.
- The list is hidden until the shortcut opens it.
