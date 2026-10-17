import { EditorView } from '@codemirror/view'

// CodeMirror mounts its base theme as an unlayered <style>, which beats every rule in the
// layered component CSS. A theme extension sits above the base theme, so the completion list
// takes the g-select menu skin here. Hosts tint it with the --control-* hooks. Selectors repeat
// the base theme's `.cm-tooltip.cm-tooltip-autocomplete` so they match its specificity and win.
const accent = (percent: number) =>
  `color-mix(in srgb, var(--control-accent, var(--secondary)) ${percent}%, var(--light))`

export const completionTheme = EditorView.theme({
  '.cm-tooltip.cm-tooltip-autocomplete, .cm-tooltip.cm-completionInfo': {
    border: '1px solid var(--lightgray)',
    borderRadius: 'var(--radius-none)',
    boxShadow: 'none',
    // Opaque, unlike the 96% select menu: the list can open over gallery images.
    background: 'var(--light)',
    color: 'var(--dark)',
  },
  '.cm-tooltip.cm-tooltip-autocomplete > ul': {
    display: 'grid',
    gap: '1px',
    maxHeight: 'min(22rem, 50dvh)',
    padding: '0.2rem',
    fontFamily: 'var(--codeFont)',
    fontSize: 'var(--text-meta, 0.75rem)',
    lineHeight: '1.5',
    fontVariantNumeric: 'tabular-nums',
    scrollbarWidth: 'thin',
    scrollbarColor: 'color-mix(in srgb, var(--gray) 45%, transparent) transparent',
  },
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li': {
    padding: '0.28rem 0.34rem',
    borderRadius: 'var(--radius-none)',
    color: 'var(--darkgray)',
  },
  // The active row is the keyboard focus: tint plus the inset ring g-select options use, so
  // the state does not rest on the tint alone.
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]': {
    background: accent(10),
    color: 'var(--dark)',
    outline: '1px solid var(--control-focus, var(--dark))',
    outlineOffset: '-1px',
  },
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li:hover:not([aria-selected])': {
    background: accent(5),
  },
  '.cm-completionMatchedText': {
    color: 'var(--dark)',
    textDecoration: 'underline',
    textUnderlineOffset: '0.2em',
  },
  // Kind icons (LSP hosts keep them; the Base query editor turns them off) read as quiet ink.
  '.cm-completionIcon': {
    width: '1.1em',
    paddingInlineEnd: '0.45em',
    color: 'var(--darkgray)',
    opacity: '1',
  },
  // The base theme draws keywords as a key emoji; the site has none.
  '.cm-completionIcon-keyword::after': { content: '"k"' },
  '.cm-completionDetail': { marginInlineStart: '0.6em', color: 'var(--gray)', fontStyle: 'normal' },
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected] .cm-completionDetail': {
    color: 'var(--darkgray)',
  },
  '.cm-tooltip.cm-completionInfo': {
    padding: '0.4rem 0.55rem',
    fontFamily: 'var(--bodyFont)',
    fontSize: 'var(--text-meta, 0.75rem)',
    lineHeight: '1.5',
  },
  '@media (pointer: coarse)': {
    '.cm-tooltip.cm-tooltip-autocomplete > ul > li': { minHeight: '44px', alignContent: 'center' },
  },
  '@media (forced-colors: active)': {
    '.cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]': {
      outline: '1px solid CanvasText',
    },
  },
})
