# Backlinks

The notes that link to this one, as a two-column table at the end of the page.

**Source:** `quartz/components/Backlinks.tsx` · styles: `quartz/components/styles/backlinks.scss`

## Preview

A copy of the server-rendered markup from `/thoughts/Attention` (selector `section.backlinks`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The component has no script; the markup is the whole component.

## Rules

- A table with the header `backlinks · description`. Each row is the linked note's title and its description, the title in a link with a popover preview on hover.
- The table sits in a bordered container, in the `meta-mono` register for the header and body type for the cells. The citation apparatus anchors to the table's end.
- A long list scrolls inside the container (`overflow`).
- Focus: the row link covers the row and draws its ring on `::after` at `-2px` (see `focus.md`, Ring placement).
