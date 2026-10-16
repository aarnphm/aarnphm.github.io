# Craft

How a Garden figure comes to explain something instead of decorating a note. The rules adapt the `anatomy` skill (`~/.claude/skills/anatomy`, its `references/craft.md`), whose figures are isometric machines. Garden figures are mostly flat, but they fail in the same ways.

## What a good figure has

1. **The mechanism, drawn.** The reader recognises the thing: the head-to-KV grouping, the tile grid walking across $QK^\top$, the page table mapping logical blocks to physical slots. Its shape comes from the system being explained, with true counts and proportions.
2. **A model behind it.** A small pure function computes every number the figure shows and every state it draws. `KVHeadGrouping` derives its cache readout from the head counts; the drawing and the readout cannot disagree because they share one source.
3. **Many small, precise parts.** Cells, slots, edges and labels that each have a job. Density comes from structure (every head, every block, every tile) rather than texture or ornament.
4. **Calm.** Thin strokes, quiet ink and one lit accent. Motion slow enough to follow.
5. **Truth in the words.** The caption states assumptions (dense BF16, weight-only model, ordinary MHA) and the readout shows units.

## Workflow

1. **Collect the truth.** Write down sizes, counts, thresholds and formulas. Cite where they come from in the note.
2. **Pick the mechanism.** Say in one sentence what moves and what that motion means: "pressing + token appends a slot; when a block fills, the page table maps a new physical block".
3. **List the parts.** Before drawing, list what is on screen and the job of each part. A part without a job is cut.
4. **Assign tones.** Map every part to a step of the tone ladder in [Palette](palette.md). Exactly one part is lit at a time.
5. **Frame it.** Controls on top, drawing in the middle, readout beside or below it (a `<dl>` card), caption under the frame.
6. **Make it live.** One state object, one render path; write attributes on refs. Ease state changes. Pause stepped tours offscreen and under reduced motion.
7. **Verify like a critic.** Zoom to 200% and 400% in the rendered page, in both themes, at 320px and desktop. Look for lines that stop short, labels that collide, colours that disagree with the caption. See [Verification](verification.md).

## Anti-patterns

Each of these makes a figure read as generated filler:

- **Boxes and arrows.** Labelled rectangles joined by arrows stand for a system without showing it. Draw the system's own units instead (heads, tokens, blocks, tiles).
- **Colour everywhere.** Every element in a different accent leaves nothing for the eye to follow. Keep structure in ink; colour the one lit part and, if needed, one second state.
- **Typed colours.** A hex value in a component stylesheet escapes the theme. It drifts when the palette changes and breaks the dark theme first.
- **Grey text.** `--gray` and accent-coloured labels read below 3:1 on paper. Inactive tabs, legends and secondary labels use `--fig-tab-fg` or `--darkgray`.
- **Glow and shadow.** `box-shadow`, `filter: drop-shadow` and inset rings simulate depth the flat grammar does not have. Mark the active node with a heavier stroke or the lit fill.
- **Rounded pills and dashed edges.** `rx` on node rects and `stroke-dasharray` on edges add a second visual language to a square, solid site. Square the rect; carry the state with tone, opacity or an empty fill.
- **Tables drawn as loose rows.** Label/value pairs floating in a padded box, separated by spacing alone, make the reader align columns by eye. Use the grid table.
- **Blueprint furniture.** Floating leader lines, numbered balloons, dimension arrows and hatching for emphasis. Labels belong on the part or in the legend and caption. (Hatching that encodes a state, such as cache-resident blocks in `KVCacheVariants`, is data, and the legend names it.)
- **Lines that stop in the air.** A connector that ends a few pixels short of its node, or a divider that stops before the frame edge, reads as broken at 2× zoom.
- **Decorative motion.** Pulses, bounces and looping shimmer with no state behind them. Motion shows a state change or a step of the mechanism.
- **Captions that disagree.** A caption naming "coral" and "olive" over salmon and sage fills, or describing a state the figure no longer has.
- **Hand-tuned numbers.** A readout typed as a string next to a drawing computed elsewhere. Derive both from the model.

## Composition

- Let the drawing fill most of the frame; keep side readouts narrow and stable (tabular numerals, fixed minimum width).
- Keep one scale across a set of related figures in a note, so the reader compares sizes honestly.
- Check the figure at 320px wide. Reflow the readout below the drawing with a container query on a descendant of the frame.
