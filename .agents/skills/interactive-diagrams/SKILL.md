---
name: interactive-diagrams
description: Build or revise registered SVG and interactive figures embedded in Garden notes.
---

# Garden figures

Use the registered components in `quartz/components/mdx`, their Preact implementation, and the shared mixins in `quartz/components/styles/figure.scss`. Keep changes scoped to the requested figure and reuse this visual grammar.

- Keep the square 1px frame and shared Flexoki accents. Reuse frame, caption, math, and foreignObject mixins. Garden figures do not use box shadows.
- Wrap interactive or large figures in `Zoomable` in the note's registered JSX fence. Let the zoom container own its presentation; restore figure padding and background without adding a border.
- Use KaTeX for mathematical labels and numeric readouts. Keep non-math request text and code-like labels as text.
- Put responsive grids on descendants of the query container. A container cannot use its own size query to restyle itself.
- Match control semantics to behavior. Use native buttons, radio inputs, or complete tab interactions as appropriate.
- Initialize page state and register cleanup within the `nav` handler. Keep computed control values distinct from shared palette tokens.
- Write a caption that explains the figure's subject and useful interaction without requiring another writing workflow.

Read only the reference needed for the change:

| Work                                                | Reference                                                      |
| --------------------------------------------------- | -------------------------------------------------------------- |
| Frame, SCSS mixins, zoom, or component registration | [Frame and registration](references/frame-and-registration.md) |
| Tabs, segments, or toggles                          | [Controls](references/controls.md)                             |
| KaTeX values, statistics, or sliders                | [Math and readouts](references/math-and-readouts.md)           |
| Hover or focus explanations                         | [Hover details](references/hover-details.md)                   |
| Source-rendered browser verification                | [Verification](references/verification.md)                     |

Reuse a shared mixin when an established pattern exists. Extract a new shared helper only when actual callers need it, and keep the change scoped to those callers.

Finish with the figure embedded and the changed behavior verified at the requested scope. For visual or interaction changes, use the verification reference; a source edit alone does not establish the rendered result.
