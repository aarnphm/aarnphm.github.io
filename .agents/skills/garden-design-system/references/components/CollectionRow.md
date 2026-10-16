# CollectionRow

A bordered list row that links to one note and shows its title, date and an optional description.

**Source:** `quartz/components/styles/listPage.scss` (`ul.section-ul`, `li.section-li`, `.note-link`).

## Inputs

- One `<ul>`; each row is an `<a href>` that covers the whole row.
- Per row: the title, the date, and an optional one-line description.

## Rules

- Use it for folder and tag pages, search results, Evergreen, and any list whose job is finding one item.
- Frame: `1px solid dark` around the list and between rows; the last row has no bottom rule.
- Title at `ui` size in `darkgray`. Date in `meta-mono`, tabular, right-aligned. Description at `meta` size on its own line.
- Keep the date slot when the date is empty (folders use `visibility: hidden`) so titles align.
- Hover: `surface` fill. Selected or matched (tag preview, keyboard cursor): `selected` fill, label `dark`.
- Focus: `2px solid focus`, offset `-2px` so the frame does not clip it.
- When a filter removes every row, hide the frame too.
- To show matches, raise matching rows to `dark`. Never fade the other rows below readable contrast.
