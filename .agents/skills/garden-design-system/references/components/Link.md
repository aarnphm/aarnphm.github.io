# Link

A link with one of two hovers: the inverted pair on prose pages, or the hover a tool page sets for itself.

**Source:** `quartz/styles/base.scss` (`a`, `a:hover`) and the overrides in `arena.scss`, `stream.scss`, `listPage.scss`, `popover.scss`, `curius.scss` and `pages/bases.scss`.

## Inputs

- An `<a href>`. Tool pages also provide the scope selector for their override.

## Rules

- Prose rest: `link` (`darkgray`), underlined.
- Prose hover: `link-hover-ink` on a `link-hover-bg` ground. Only on reading pages: notes, letters, technical pages. A link that wraps a code figure or an image drops the ground.
- Tool text link: no ground; ink rises to `dark`; underline.
- Tool row link: the row fills with `surface` (`arena-surface` in Arena); the label keeps its colour.
- Tool block link (a grid cell): `surface` fill; the border rises to `gray`.
- Scope each tool override to its page (for example `[data-slug*='arena']`) and set `background: transparent` on hover, so the prose inversion never reaches a tool.
- Every form uses the `focus` ring.
