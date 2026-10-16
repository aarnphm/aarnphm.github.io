# ArenaReader

The Arena reader frame: a mono header bar with a source link and actions, over a long-form reading pane.

**Source:** `arena-feed.scss` (`.arena-reader`, `.arena-reader-header`, reader `button`).

## Inputs

- The source link.
- Actions as native buttons, with `aria-pressed`, `aria-current` or `aria-expanded` to match their behaviour.
- The block's text.

## Rules

- Column `width-arena`, `arena-line` on both sides, `100dvh`. Each pane scrolls on its own.
- Header: mono at 0.66rem, `arena-line` bottom rule, minimum height `control-reader` + 0.5rem + 1px, `space-4` gutters that respect the safe area.
- Buttons: `control-reader` (1.5rem) high, padded `0.22rem 0.7rem`, `arena-line` border, no fill, tabular figures.
- Button hover: `arena-surface` fill, `gray` border. Pressed, current or expanded: `arena-selected` fill, `dark` label. Disabled: opacity 0.5.
- Body: `reader` style (1rem / 1.65).
