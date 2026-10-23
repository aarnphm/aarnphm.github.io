# Palette

The command palette: a full-screen search over commands and notes, opened from the keyboard.

**Source:** `quartz/components/Palette.tsx` · styles: `quartz/components/styles/palette.scss`

## Preview

A copy of the server-rendered markup from `/` (selector `.palette`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The container shows open (`class="active"`) without its `popover="manual"` attribute. The preview is static because it fetches the site's content index.

## Rules

- On the site the container is a manual popover that starts hidden. Results and commands fill in at run time.
- The field row holds the input and a `pagerank ↗` button. A thin progress bar under it shows while the semantic index loads.
- The helper line lists keys (`↑↓`, `↵`, `⟶`, `⌘ ⌥ ↵`, `esc`) and the filters `images · pdf · code · py · all`.
- Focus: the input is the only control focus lands on while the overlay is open, so it shows the caret and the open overlay (see `focus.md`, item 7). The `pagerank` button keeps its ring.
