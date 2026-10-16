# Button

A square toolbar button with a text label and an optional 16px icon.

**Source:** `quartz/components/styles/baseViewSelector.scss` (`.text-icon-button`).

**Corrections:** the source focus ring is sage (1.51:1 on paper). This rendition uses `focus`.

## Inputs

- A native `<button type="button">` and its label.
- An optional inline SVG icon in `currentColor` with `aria-hidden="true"`.
- `aria-expanded` when the button opens a menu.

## Rules

- Rest: `light` fill, `1px solid lightgray` border, label `dark` at 500.
- Hover, fine pointers only: `highlight` fill, `gray` border.
- Expanded: `highlight` fill, `dark` border.
- Focus: `2px solid focus`, offset 2px.
- Disabled: native `disabled`, opacity 0.5.
- Height `control-base` (2.125rem); 44px under `pointer: coarse`.
- Square corners. No accent fill at rest. A press may scale to 0.97 briefly; nothing else moves.
