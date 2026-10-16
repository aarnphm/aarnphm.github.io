# SearchField

A square search input that filters the list or grid beside it.

**Source:** the Base search bar (`baseViewSelector.scss`, `#base-search-bar`) and the Arena search.

**Corrections:** both sources draw focus as a sage border with no outline (1.51:1). This rendition uses the `focus` ring.

## Inputs

- An `<input type="search">` with a visible label or an `aria-label`.
- Placeholder text that names what it searches.
- A live result count or an empty state nearby.

## Rules

- Height `control-base` (2.125rem), `1px solid darkgray`, square, text `dark` at `ui` size. 16px below `bp-mobile` so iOS does not zoom.
- Placeholder in `darkgray`.
- With no results, say so and offer to clear the query.
