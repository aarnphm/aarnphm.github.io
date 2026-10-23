# Search

The search button and its open panel: a text field, a lexical or semantic mode toggle and a result list.

**Source:** `quartz/components/Search.tsx` · styles: `quartz/components/styles/search.scss`

**Corrections:** in the source, the inactive mode icons, the status text and the button label are `gray`, and the active icon is paper on `secondary` (1.4:1 in the light theme). This rendition uses `darkgray` for the first three and `on-accent` for the active icon.

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `.search:not(.bases-search)`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The panel shows open (`class="active"`) without its `popover="manual"` attribute. The preview is static because it fetches the site's content index.

## Rules

- Opens from the button or the keyboard shortcut; Escape closes it.
- The mode toggle is a radio group of buttons with `aria-pressed`; the active one takes the `secondary` fill with an `on-accent` icon. A preview toggle opens the result preview beside the list.
- The status text (`semantic-status`) is empty while idle and names the semantic index state when it loads.
- The helper line lists the keys, including `⌥⇧S` for semantic search and `⌥P` for the preview.
- Focus: the input is the only control focus lands on while the panel is open (see `focus.md`, item 7).
