# ArenaControls

The Arena channel control row: a search input that takes the free width, and a mono view toggle.

**Source:** `arena.scss` (`.arena-channel-controls`, `.arena-search-input`, `.arena-view-toggle`, `.arena-view-toggle-button`).

**Corrections:** the source search sets `outline: none` and shows a sage border on focus (1.51:1), and its toggle labels are `gray`. This rendition uses `focus` and `darkgray`.

## Inputs

- A search input with an accessible name.
- Toggle buttons with `aria-pressed`, each with a 14px icon and a lowercase label.

## Rules

- Height `control-collection` (1.75rem); 44px under coarse pointers. Gap 0.375rem.
- Search: mono at 0.7rem, `arena-line` border, room on the right for the icon. 16px on mobile.
- Toggle: buttons inside one `arena-line` frame, separated by `arena-line` rules. Active: `arena-selected` fill, `dark` label.
