# Focus

Every focusable control shows where keyboard focus is. The default indicator is the ring: `outline: 2px solid` `focus`, `outline-offset: 2px`. This section says when a page may remove the browser outline (`outline: none`) and what must take its place.

The site has no global `:focus-visible` rule for buttons and links. When a rule removes the outline, nothing else draws focus unless the same component draws it. So every `outline: none` comes with its replacement in the same component.

## Ring placement

- Outside the box: offset `2px`. This is the default.
- Inside a frame: offset `-2px`. Use it in framed lists, in `overflow: hidden` or `overflow: auto` containers, and next to sticky cells, where an outside ring is clipped (backlinks table, Arena blocks, base kanban links, `.table-container`).
- The ring keeps its colour and width when the control is selected, pressed or expanded. A ring is the only state that sits on top of the `selected` fill.
- In forced colours, the ring uses `Highlight`.

## Allowed replacements

Remove the outline only when one of these draws focus instead.

1. **Ring on a pseudo-element.** A row link that covers the whole row with `::after { inset: 0 }` puts the ring on `::after` at `-2px` (backlinks, Evergreen). The ring then frames the row, not the title text.
2. **Ring on a child.** When the focusable element is a wrapper and a child is the visible box, the child takes the ring (`.slides-toc-label`, the workout lap bar, the DEXA column bar).
3. **Ring on the field frame.** A text input inside a framed field removes its own outline. The frame takes a 2px `focus` ring through `:has(> input:focus-visible)` or `:focus-within` (page-list search, the Base query bar). The frame is what the reader sees as the control.
4. **Stroke on a chart mark.** A focusable SVG mark or chart region shows focus with a 2px `dark` stroke on the mark or a frame rectangle (data chart hit marks, pole response handle, head spectrum frame). A readout or an active label is state, not focus, and does not count.
5. **Fill and text.** The bookcase uses no ring, by decision. A focused topic shows a 2px `currentColor` bar under its label (on `::after`, because the label clips a text underline); an unselected focused topic also takes a `lightgray` fill with `dark` text. The selected topic keeps its `dark` fill with `light` text, so focus and selection stay distinct. Other surfaces use fill and text only when the result differs from both hover and selected.
6. **Programmatic targets.** A `<dialog>`, a `tabindex="-1"` region or a keyboard scroll scope that script focuses needs no ring, because the reader did not tab to it. The controls inside it keep their rings.
7. **The only input in an open overlay.** The search modal, the palette, the triathlon command line and the notebook vim command line show focus with the caret and the open overlay itself. This applies only while the overlay is open and Escape closes it.

## How to write it

- Remove the outline in the same rule that draws the replacement, on `:focus-visible`. If mouse focus must stay quiet, use `:focus:not(:focus-visible)`.
- Never write `:focus { outline: none }` or `outline: none` with no pseudo-class on something a reader can tab to, unless the same component also draws one of the replacements above.
- Put the ring on the same selector as the reset, or on a stronger one. A scoped rule such as `.scope :where(a, button, [tabindex]):focus-visible` has the specificity (0,2,0). A later `.x:focus { outline: none }` has the same weight and wins, so the ring never shows.
- An `outline` declaration that sets only a colour draws nothing, because `outline-style` stays `none`. Write all three parts: `2px solid var(--dark)`.
- A region a reader reaches with Tab (`tabindex="0"`, for example a scroll region with an `aria-label`) is not a programmatic target. It needs a ring.
- Do not use `--shadow-focus` or an inset `box-shadow` as a ring. The system has no shadows.

## Tabs

With a roving tabindex, focus always lands on the selected tab. So a focus style that equals the selected style is invisible. Draw the ring over the selected fill. `triathlon-best-efforts.scss` is the reference fix.

On a solid accent fill (`tri-accent`, `dark`), put the ring outside the tab at `2px`, on paper. An inset ring on the fill falls under 3:1 on some accents in the dark theme. When the tablist scrolls (`overflow-x: auto`), give it `padding: 4px` and `margin: -4px` so the outside ring has room and the layout does not move (`.tri-map-tab`). A tablist that clips and has no accent fill keeps the `-2px` ring (`.notebook-output-tab`).

## Colour

- `focus` (`dark`) everywhere. Triathlon panels may use `tri-be-focus`.
- An authored page (the 404 scene) may use its own palette for the ring, at 3:1 or better against its ground.
- Debt in the current code: `secondary` rings (Arena, the PDF reader, zoomable images), 1px rings in `tri-accent` at 68% on triathlon charts, the `darkgray` ring on the triathlon workspace, and 1px border swaps to `secondary` on framed inputs (Arena search, stream search, password inputs, the analytics search, the PDF embed). `secondary` reads 1.51:1 on paper.

## Known gaps

None known (2026-10-22). The seven gaps listed earlier (PDF reader scroller, selected `.tri-map-tab`, `.tri-note-foot` triggers, power radar, are.na title link, mobile search button, bookcase topics and notebook output tabs) now draw a replacement, checked by keyboard in the light and dark themes.
