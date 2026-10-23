# Button

A square toolbar button with a text label and an optional 16px icon.

**Source:** `quartz/components/styles/baseViewSelector.scss` (`.text-icon-button`, with `--base-query-line`, `--base-query-surface` and `--base-query-selected` from `baseQuery.scss`).

## Inputs

- A native `<button type="button">` and its label.
- An optional inline SVG icon in `currentColor` with `aria-hidden="true"`.
- `aria-expanded` when the button opens a menu.

## Rules

- Rest: `light` fill, `1px solid` border in `gray` at 55%, label `dark` at 500, icon `gray`. The border keeps its colour in every state, so a button, the search field and the filter pair beside it share one line.
- Hover, fine pointers only: the `surface` fill (`light` 86% over `lightgray`).
- Expanded (open menu): the selected fill (`secondary` 18% over `surface`).
- Focus: `2px solid dark`, offset 2px.
- Disabled: native `disabled`, opacity 0.5.
- Height `control-base` (2.125rem); 44px under `pointer: coarse`.
- Square corners. No accent fill at rest. A press may scale to 0.97 briefly; nothing else moves.
