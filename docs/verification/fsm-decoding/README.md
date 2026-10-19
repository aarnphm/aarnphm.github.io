# fsm-decoding

## Change

`FSMDecoding` (`quartz/components/mdx/FSMDecoding.tsx`, `quartz/components/styles/fsmDecoding.scss`, `quartz/components/scripts/fsm-decoding.inline.ts`) replaces five image embeds:

- `content/thoughts/structured outputs.md`: `constrained-json-fsm.webp` (Method 1), `jump-forward-decoding-fsm.webp` (Method 3), `part-of-json-fsm.webp` and `compressed-fsm-json.webp` (Coalescence, one `view="graph"` figure), `fsm-iterative-generations.webp` (moved to just after the `finite automaton` callout, because no note nests a jsx fence in a callout).
- `content/posts/structured decoding.md`: `constrained-json-fsm.webp`.

The component compiles the regex to a minimal DFA (Thompson NFA, subset construction, Moore minimisation, BFS numbering). It derives masks, token targets, the compressed graph, the jump-forward frames and every readout from that DFA and the listed toy vocabulary. The inline script only applies the precomputed frames.

`json-difference-in-sampling-distribution.webp` stays. Its bars have no values in the source, and the effect (a different segmentation changes the model state) needs a model to compute. Any numbers would be invented.

## Replay

```
node docs/verification/fsm-decoding/replay.mjs
```

It reads the five `<FSMDecoding ... />` tags from the two notes, renders them through `docs/verification/figure-harness.mjs`, and also runs both notes through the Quartz `Codeblock` transformer (`pipeline.mjs`) to confirm each embed parses with its props.

Checks, for every frame of every mode, with clicks, Enter and Space, and ArrowRight on the radio group:

- drawn node and edge states equal the frame model, with one current state that the action text names;
- the mask equals an oracle built from the language itself (the four JSON strings, or the number regex, which is prefix-closed), and every allowed token shows its target;
- the sampled token is lit in mask and tape with the right pass tag; readout passes and allowed counts match the drawing;
- button disabled states, focus moving to back when step disables, back and reset;
- final outputs are in the language; 10 passes for the character FSM, 2 for jump-forward, 4 for the number regex;
- the compressed edge table spells exactly the four accepted strings;
- light, dark and 320px screenshots, no horizontal scroll at 320px, no page errors.

## Results

644/644 checks passed (`artifacts/results.json`). Screenshots: `artifacts/{light,dark,320}-{0..4}.png`, `artifacts/char-step-4.png`, `artifacts/jf-step-2.png`.

## Not checked

- No dev watcher was running, so the figure was not inspected inside the full Quartz page or after SPA navigation.
- 200% and 400% browser zoom were not captured; the Zoomable wrapper was rendered but not opened.
- The toy vocabulary and the greedy split of forced text are illustrations, not a real tokenizer.
