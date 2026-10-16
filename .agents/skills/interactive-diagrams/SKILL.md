---
name: interactive-diagrams
description: "Build, revise or review the SVG and interactive figures embedded in Garden notes (quartz/components/mdx). Covers the mechanism-first craft, the figure tone ladder in Garden tokens, controls, KaTeX readouts and verification. Use it for any explanatory diagram on the site, including isometric figures built with the anatomy kit."
---

# Garden figures

A Garden figure explains one mechanism from a note: a KV cache, a tile loop, a page table, a decision tree. The reader learns the idea because the figure draws the mechanism itself, with true counts, and lets them move it. Everything else in this skill serves that.

The figures live in `quartz/components/mdx` (Preact), style through the mixins in `quartz/components/styles/figure.scss`, and sit inside the Garden design system. Read the `garden-design-system` skill for tokens and page registers; this skill covers what is particular to figures.

## The rules that decide most of the quality

These come from the `anatomy` skill's craft, translated from isometric objects to the flat diagrams this site uses. `references/craft.md` gives the reasoning and the failures behind each one.

1. **Draw the mechanism.** Draw the real structure with its real counts: eight query heads sharing two KV heads, an eight-slot block pool, the actual tile grid. Boxes labelled "Encoder" joined by arrows explain nothing. If the idea has a physical analogue that maps one-to-one (a lock, a drawer chest, a plotter), an isometric object can carry it; see `references/isometric.md`.
2. **Truth first.** Write the numbers and behaviour down before any geometry. One pure function or module computes the state; the drawing, the readout and the caption all read from it. A readout that disagrees with the drawing is a bug.
3. **One thing lit.** Structure is quiet ink; the part the reader should watch now is the only `--fig-salmon` fill. `--fig-sage` marks the second state (done, kept, shared). A third colour needs a reason the caption can name. See `references/palette.md`.
4. **Every colour is a token.** No hex, `rgb()` or `oklch()` in a figure stylesheet. Component tokens (`--kvg-*`, `--fat-*`) are defined from `--fig-*` and theme variables, with fills mixed into `--light` (light) or `--lightgray` (dark).
5. **Text is ink.** Labels and values use `--dark`; card headings `--fig-note`; captions `--fig-caption-fg`; tab and secondary labels `--fig-tab-fg`. `--gray` and the accents never carry text in the light theme.
6. **Lines resolve.** Every edge, connector and rule ends on a node, a cell or the frame. No floating leader lines, balloons, dimension arrows or dashed hidden edges. Math labels sit on the part they name.
7. **Flat and square.** 1px frame, square corners, no `box-shadow`, no `filter: drop-shadow`, no glow, no gradient, no decorative left border. A state change is a fill, a stroke width or ink weight.
8. **Calm motion.** State changes ease over 120 to 220ms on named properties. Stepped animations run only while visible and stop under `prefers-reduced-motion`, leaving the final state drawn.
9. **Real controls.** Native buttons, radio groups and tablists with the full keyboard contract; visible focus in `--dark`. Listeners and cleanup register inside the `nav` handler.
10. **Caption tells the truth.** It states what the figure shows, its assumptions, and what to do. Colour words name the palette (salmon, sage), matching what the reader sees.

## Read only the reference the change needs

| Work | Reference |
| --- | --- |
| Designing a new figure, or a figure that reads as slop | [Craft](references/craft.md) |
| Any colour, fill, stroke or text ink | [Palette](references/palette.md) |
| Frame, SCSS mixins, zoom, registration | [Frame and registration](references/frame-and-registration.md) |
| Tabs, segments, toggles | [Controls](references/controls.md) |
| KaTeX labels, statistics, sliders | [Math and readouts](references/math-and-readouts.md) |
| Hover or focus explanations | [Hover details](references/hover-details.md) |
| A physical-object figure with the anatomy kit | [Isometric](references/isometric.md) |
| Calling it done | [Verification](references/verification.md) |

`quartz/components/mdx/KVHeadGrouping.tsx` with `kvHeadGrouping.scss` is the reference figure: model-driven readout, pre-rendered KaTeX states, tokens defined from `--fig-*`.

Reuse a mixin when an established pattern exists. Extract a shared helper only when actual callers need it. Finish with the figure embedded and its changed behaviour checked in the rendered page, in both themes; a source edit alone does not establish the result.
