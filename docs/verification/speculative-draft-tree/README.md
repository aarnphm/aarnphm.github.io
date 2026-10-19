# SpeculativeDraftTree verification

## Change

`content/thoughts/Speculative decoding.md` replaces two EAGLE paper images with one figure, `SpeculativeDraftTree`:

- `eagle-feature-prediction-one-time-step.webp` (EAGLE figure 3, feature uncertainty), removed from the EAGLE-1 section.
- `eagle-figure-5-comparison.webp` (EAGLE figure 5, drafter comparison), replaced in place under "architecture".

The figure is a stepper over one round after the target sampled "I" from "How can". A drafter selector (small LM, Medusa, EAGLE, EAGLE-3) and a step selector (draft 1 to 3, verify, accept) drive one pure model in `quartz/components/mdx/SpeculativeDraftTree.tsx`. The model builds each draft tree, the tree attention mask, the test log and the readout. The inline script only applies the precomputed states.

All probabilities are synthetic. Each drafter's $q$ is $(1-\lambda)p + \lambda n$ per depth, with Medusa's heads using the marginal over depth-$k$ contexts. Verification uses $U \le \min(1, p/q)$ with $U = 0.7$ fixed. A rejected child turns $p$ into the normalized residual and is dropped from $q$ before the next sibling. Residual and bonus draws are fixed to the most likely token.

Images kept: `eagle-figure-6-architecture.webp` and `eagle-3-inference-pipeline.webp`. They show frozen target modules, the FC and decoder layers, and the feedback wiring. The figure does not draw these.

## Replay

```sh
node docs/verification/speculative-draft-tree/replay.mjs
```

The replay needs the shared harness `docs/verification/figure-harness.mjs` and Chromium at `/opt/pw-browsers/chromium` (or `CHROMIUM`). It renders two instances inside `Zoomable`. Instance one uses the caption from the note.

## Results

252 of 252 checks pass. Results are in `artifacts/results.json`, with screenshots in `artifacts/`.

The checks are, for each of the 20 drafter and step states:

- The visible rule, strip and log match the selected state.
- The drawn draft nodes equal the "drafted tokens" readout.
- The newest depth is lit while drafting.
- The draft and target pass counts are correct.
- Each mask row shows the cached prefix, its ancestors (found from tree geometry) and itself, and the lit rows are correct.

At accept, the replay also checks these items:

- The accepted nodes form a path from the root.
- The output strip is the prefix, the accepted path and one target token.
- The emitted and plain-decoding counts agree.
- Each logged test agrees with $\min(1, p/q)$ and $U = 0.7$, recomputed from its logged $p$ and $q$.

Other checks:

- Medusa gives both depth-1 nodes the same children and $q$.
- EAGLE and EAGLE-3 give the two branches different children.
- Arrow keys and Tab move through both radio groups, and focus is visible.
- The second instance stays independent.
- Cleanup removes the listener.
- Light, dark and 320px screenshots are taken.
- The page does not scroll horizontally at 320px, and the page has no errors.

Accept states: small LM 1 of 3 accepted (emits "help you"), Medusa 3 of 3 ("help you to find"), EAGLE 2 of 3 ("help you with"), EAGLE-3 3 of 3 ("help you with this").

## Not checked

- The page was not built through the Quartz dev watcher, because no watcher was running. The replay renders the component through the same esbuild, Sass and inline-script loaders.
- SPA navigation between pages.
- 200% and 400% browser zoom.
