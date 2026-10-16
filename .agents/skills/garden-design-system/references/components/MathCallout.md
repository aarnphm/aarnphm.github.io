# MathCallout

A framed box for a definition, theorem, lemma or proof, with an optional two-column proof.

**Source:** `quartz/styles/custom.scss` (`blockquote.callout.callout-math`, `.math-proof*`) and `quartz/plugins/transformers/ofm.ts` (proof rows).

**Corrections:** the source draws proof reasons in `gray` (2.00:1 on paper). This rendition uses `darkgray`.

## Inputs

- A callout of type `definition`, `theorem`, `lemma`, `proposition`, `corollary`, `proof`, `claim`, `axiom`, `conjecture` or `math`, with a title.
- For a two-column proof: a line that reads "Proof." and then one paragraph per step. A trailing `[reason]` on a step moves into the reason column.

## Rules

- Frame: `1px solid math-frame`, no fill, square, padded `0 space-3`.
- Title: `body` font, italic, 600, `dark`. A proof title is 400.
- Display math inside gets `space-4` above and below.
- Proof rows: step and reason on a grid, the reason column at least 12rem. Below 700px the reason stacks under its step.
- Reasons at 0.95em in `darkgray`.
