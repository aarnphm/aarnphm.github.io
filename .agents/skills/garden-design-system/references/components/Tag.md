# Tag

A square, outlined mono label that links to a tag page.

**Source:** `listPage.scss` (`.tag-highlights li`) and the `tagLabel` mixin.

## Inputs

- The tag text, lowercase as written in frontmatter, and its `/tags/<name>` link.

## Rules

- `1px solid gray` border, square, `meta-mono` text in `darkgray`.
- Hover: ink `dark`, border `darkgray`.
- Tags are links, so they take the `focus` ring.
- Hide tag highlights in popovers, in stacked notes and at tablet width, where they crowd the row.
