# Completion

The completion list under a code cursor: the Base query editor, the markdown editor and the notebook code editor with its language server (Pyright, Lean).

**Source:** `quartz/util/codemirror-completion-theme.ts` (`completionTheme`), a CodeMirror theme extension. Hosts add it next to `autocompletion()`: `quartz/components/scripts/base-query-editor.ts`, `quartz/components/scripts/markdown-editor.ts` and `quartz/runtime/editor/code-editor.ts` (every notebook cell, whatever its language bridge).

**Skin:** the `g-select` menu, set in code. CodeMirror mounts its base theme as an unlayered `<style>` that beats layered component CSS, so the skin lives in a theme extension whose selectors repeat `.cm-tooltip.cm-tooltip-autocomplete`. Do not restyle the list from a page stylesheet: a deeper page selector wins over the theme and splits the look.

## Anatomy

- Frame: 1px `lightgray`, `radius-none`, no shadow, opaque `light` ground (the list can open over images).
- List: `code` family at `meta` size, line height 1.5, rows 1px apart, `0.2rem` inset, at most `min(22rem, 50dvh)` tall with a thin scrollbar.
- Row: `darkgray` label, matched characters `dark` and underlined, detail `gray` after `0.6em`.
- Active row (keyboard focus): `selected` ground (accent 10% into `light`), `dark` label, 1px `focus` ring inset by 1px. Hover on another row: accent 5%.
- Kind icon: shown by language-server hosts, hidden in the Base query editor (`icons: false`). `darkgray`, `1.1em` wide.
- Info panel (`.cm-completionInfo`): the same frame, `body` family, `0.4rem 0.55rem` padding.

## Host hooks

- `--control-accent` tints the active and hover rows (default `secondary`). The notebook sets it to `--notebook-active-green` on `.notebook-code-cell`.
- `--control-focus` sets the active-row ring (default `dark`).

## Neighbouring tooltips

Language-server hover, signature help and lint tooltips in the notebook use the same panel from `quartz/styles/pages/notebook.scss`: 1px `lightgray`, `light` ground, `code` at `meta`. Documentation text is `darkgray`. The active parameter in a signature is `dark`, bold and underlined; sage would read 1.5:1.

## Rules

- ArrowUp and ArrowDown move the active row, PageUp and PageDown move by a page, Enter or Tab accepts, Escape closes. Ctrl-Space (Alt-` on macOS) opens the list on demand.
- The active row keeps the ring as well as the tint, so the state never rests on colour alone. Under `forced-colors` the ring is `CanvasText`.
- Under `pointer: coarse` rows are 44px tall.
