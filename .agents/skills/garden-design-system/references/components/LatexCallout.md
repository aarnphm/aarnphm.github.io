# LatexCallout

A numbered theorem, lemma or proof from a course's converted LaTeX notes.

**Source:** `quartz/plugins/transformers/ofm.ts` (callouts, `data-callout-number`) · `quartz/scripts/vendor-ocw-latex.ts` (converter) · styles: `quartz/styles/custom/callouts.scss` (`blockquote.callout.callout-math`)

## Preview

A copy of the server-rendered markup from `/courses/18-156-projection-theory-spring-2025/notes/16` (selector `blockquote.callout`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static.

## Rules

- This is the MathCallout pattern as the site renders it, with the `latex` metadata (`> [!lemma|latex] 16.1`). The title carries the number and the block has an anchor id (`lemma-16-1`) that `\ref` links resolve to.
- A math-bearing title keeps only the number in the header and moves the name to the first body line.
- The last line links the page of the lecture PDF the statement came from (`p. 1`) and the course statements index (`statements`), separated by `·`.
- Proofs (`[!proof|latex]+ of lemma 3.2`) fold; the unit bar's `fold proofs` button folds them all.
- Display math and tables inside scroll sideways within the frame (see MathCallout).
