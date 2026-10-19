# crosscoders: cross-layer feature figures

## Change

Two figures replace four of the eight concept diagrams in `content/thoughts/sparse crosscoders.md`. Each caption says it follows Anthropic's crosscoders figure.

| Image                                                | Figure                                |
| ---------------------------------------------------- | ------------------------------------- |
| `proposal-formed-by-cross-layers-superposition.webp` | `CrossLayerFeatures view="lifecycle"` |
| `dedup-features-persistent.webp`                     | `CrossLayerFeatures view="circuit"`   |
| `crosscoder-setup.webp`                              | `CrosscoderMasks dims="coder"`        |
| `crosscoders-variants.webp`                          | `CrosscoderMasks dims="variant"`      |

The image files stay on disk. These four stay in the note: `additive-residual-stream-llm.webp`, `feature-neurons.webp`, `one-step-circuit.webp` and `parallel-joint-branch.webp`. They redraw one graph in an equivalent form, or show neurons in superposition. They have no count or rule that a model could compute.

- Lifecycle: a synthetic profile, shaped after the preview's lifecycle figure, over 8 blocks. Forming and amplifying write $+1$, and deletion writes $-s$. $s_{l+1}=s_l+w_l$ gives $0,0,0,1,2,2,3,3,0$. With 8 per-layer SAEs the feature is 5 latents. With a crosscoder it is 1 latent with $z=1$ and $\|d^l\|=s_l$. Both have an $L_1$-of-norms penalty of $11\lambda$.
- Circuit: the preview's example. Features 1 and 2 are present by $L$, MLPs in layers $L+2$ and $L+3$ make feature 3, and all persist to $L+4$. Per-layer SAEs give 13 latents and 14 edges (10 persistence, 4 formation). The crosscoder gives 3 latents and 2 edges.
- Masks: 5 residual layers. A grid of read layer against reconstructed layer comes from each rule. The rules are autoencoder ($j=i$), transcoder ($j=i+1$), acausal (all), weakly causal ($j\ge i$) and strictly causal ($j>i$). Locality can be global, a local window of $K=2$, or every other layer.

## Replay

```sh
node docs/verification/crosscoders/replay.mjs
```

The replay needs `node_modules` at the worktree root and Chromium at `/opt/pw-browsers/chromium` (or `CHROMIUM`). It uses `docs/verification/figure-harness.mjs`. The output goes to `artifacts/<figure>/`: `page.html`, `results.json`, light, dark and 320px screenshots, and one screenshot for each state.

## Results

32 of 32 checks pass.

- CrossLayerFeatures, 22 checks. The replay reads the stage words from the drawing and computes the writes again. It then checks the block values, the sage blocks, the strength labels and bars, the salmon latents, the crosscoder span and the decoder norms. It also checks the readouts and the $11\lambda$ penalty. The circuit nodes are the 13 labels 1a to 3c. The drawn edges are found from their endpoints, and they equal the 14 edges that the replay builds from the preview's text. The crosscoder view has 3 latents and the edges 1>3 and 2>3. Every line ends on a node. The radio groups work with the pointer and with the arrow keys, and they show a focus ring. The checks also cover page errors and horizontal scroll at 320px.
- CrosscoderMasks, 8 checks. The replay restates the rules. For 3 coders and 9 variants, each with 5 read layers, it checks every grid cell, the box count, the lit boxes, and which stations the lit edges reach. It also checks where the arrowheads are, which stations are hollow, the rule text and the readout. The pair counts are 25, 13 and 9 for acausal, 15, 9 and 6 for weakly causal, and 10, 4 and 3 for strictly causal. The checks also cover the keyboard, focus, page errors and scroll at 320px.
- Note, 2 checks. The four fences parse with remark-mdx, and each caption gives the attribution. The four kept images are still in the note, and the DataChart embed did not change.

`oxfmt` and `oxlint` are clean. `tsgo --noEmit` shows only the 18 baseline errors, and none of them are in these files.

## Not checked

- The full Quartz build and the dev watcher, because neither runs in this environment. The replay parses the fences but does not render them through `codeblock.ts`.
- 200% and 400% browser zoom. The SVGs scale with their viewBox. At 320px the CrossLayerFeatures drawing scrolls sideways inside its frame, and the CrosscoderMasks panels stack.
- SPA navigation cleanup across pages.
