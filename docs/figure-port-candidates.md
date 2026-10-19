# Figure port candidates

Survey of image embeds in notes (2026-10-19) that a Garden figure can replace. Scope: `content/` except `content/courses/`. Photos, artworks, slide photos, assignment outputs and paper figures kept for provenance stay as images.

Ranked by images removed per component and by how much an interaction teaches. "Reuse" means an existing component in `quartz/components/mdx` already draws the mechanism.

## Done

| Note                         | Images                           | Component            |
| ---------------------------- | -------------------------------- | -------------------- |
| `thoughts/KV compression.md` | `pyramid-kv.webp`, `ada-kv.webp` | `KVBudgetAllocation` |

## Tier 1: one component replaces many images

| Note                                                                                                                                      | Images                                                                                                                                          | Proposed figure                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lectures/412/notes.md`                                                                                                                   | 14 matplotlib plots: `Qwen3-0.6B_layer*_head*` spectrum and heatmap pairs                                                                       | `HeadSpectrum`: pick layer and head; log singular values, cumulative energy $E_k$, $r_{\text{eff}}$ and the 10-token attention map from one record. Needs `latent_projection.py` to also write JSON.              |
| `thoughts/structured outputs.md`, `posts/structured decoding.md`                                                                          | 8 + 4 (two shared): `constrained-json-fsm`, `jump-forward-decoding-fsm`, `part-of-json-fsm`, `compressed-fsm-json`, `fsm-iterative-generations` | `FSMDecoding`: step a regex FSM token by token, show the allowed-token mask, toggle jump-forward compression.                                                                                                     |
| `university/.../sfwr-4aa4/system response.md`, `sfwr-3dx4/Time response.md`, `sfwr-3dx4/Root locus control.md`, `thoughts/Z-transform.md` | 8 + 2 + 4 + 1: pole plots, step responses, same envelope/frequency/overshoot, `s-plane-stability`                                               | `PoleResponse`: drag a pole pair in the s-plane; draw $c(t)$ with $T_r$, $T_p$, $T_s$ and %OS. Constrain the drag to $\sigma$, $\omega_d$ or $\zeta$ to get the three "same" plots. A root-locus mode sweeps $K$. |
| `thoughts/KV compression.md`                                                                                                              | `kv-compress-vllm`, `paged-attention-block-kv-compress`                                                                                         | Extend `PagedKVTable` to per-head block tables, with the 10-step eviction of KV-Compress as a stepper.                                                                                                            |

## Tier 2: one image, high teaching value

| Note                                                                            | Image                                                                             | Proposed figure                                                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `thoughts/tsfm/tangents-3.md`                                                   | `all-reduce-meta`                                                                 | 4-GPU all-reduce = reduce-scatter then all-gather, stepped. Check `RingRotation` and `TreeReduction` first.                              |
| `thoughts/DS32.md`, `lectures/430/notes.md`                                     | `DSA`, `indexer-dsa-hf`, `nsa-infllm-dsa-dinstinction`, `native-sparse-attention` | Sparse-attention mask figure: compressed, selected and sliding branches (NSA) and indexer top-$k$ (DSA). Start from `SlidingWindowMask`. |
| `thoughts/Speculative decoding.md`                                              | 4 EAGLE figures                                                                   | Draft-tree stepper: draft, verify, accept prefix.                                                                                        |
| `thoughts/quantization.md`                                                      | `quantisation-format`, `compute-flow-mxformat`                                    | Bit-field figure for FP32/FP16/FP8/INT8 with value decode; MX block scale.                                                               |
| `university/.../sfwr-3db3/Transaction.md`                                       | `venn-schedule`, 3 schedule tables                                                | Schedule builder that classifies serial, conflict-serializable, recoverable, ACA, strict.                                                |
| `university/.../sfwr-4aa4/PIP and PCP.md`                                       | `PIP`, `PCP`                                                                      | Priority inheritance and ceiling timeline simulator.                                                                                     |
| `thoughts/sparse autoencoder.md`                                                | `gated_jump_relu`, `gated-sae-architecture`                                       | JumpReLU plot with a $\theta$ slider; gated path.                                                                                        |
| `thoughts/optimization.md`                                                      | `convergence-vs-step-side-momentum`                                               | Rate $\lvert 1 - \eta\lambda \rvert$ against step size, with a momentum slider.                                                          |
| `university/.../sfwr-2c03/Graphs.md`                                            | `example-graph-dfs`                                                               | DFS stepper on the same graph.                                                                                                           |
| `university/.../sfwr-4ml3/Convolutional Neural Network.md`                      | `max-pooling`, `sparse-connectivity-images`                                       | Pooling window walk; receptive field.                                                                                                    |
| `university/.../sfwr-3dx4/Block Diagrams.md`, `sfwr-4aa4/closed loop system.md` | 2 + 5 block diagrams                                                              | Block-diagram reduction stepper.                                                                                                         |

## Tier 3: charts and tables

- Bar or line charts with numbers in the image: `posts/structured decoding.md` (`vllm-new-xgrammar`, `vllm-xgrammar-decode-time-per-output-token`), `thoughts/sparse crosscoders.md` (`all-layer-crosscoder-vs-per-layer-sae`), `thoughts/tsfm/tangents-3.md` (`comparison-ddp-fsdp`), `thoughts/economies of scale.md` (LRAC curve).
- Tables as images that need Markdown, not a component: `sfwr-2c03/Sorting.md`, `sfwr-3dx4/steady-state error.md`, `eng-3px3/Conversion Factors.md`, `sfwr-3db3/Relational Algebra.md`, `sfwr-3db3/Design theory.md`.

## Partial reuse

These existing components draw a related mechanism. Each needs a check against the note before it replaces the image:

- `thoughts/Radix tree.md` `Patricia_trie.svg`: `RadixPrefixTree` draws a token prefix tree, not a string Patricia trie.
- `thoughts/DS32.md` `mqa-mha-mode`: `MLALatentPath` draws the latent path, not the MHA/MQA absorption modes.
- `thoughts/FFN.md` `residual-network`: `ResidualStream` draws the transformer stream, not a ResNet block.
