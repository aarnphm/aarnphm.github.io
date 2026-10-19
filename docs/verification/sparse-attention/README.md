# sparse-attention

## Change

- `SparseAttentionMask` draws a causal mask of 64 toy tokens with five patterns: NSA compressed, selected and sliding branches, their gated union, and DSA top-k. A strip above the grid shows how the chosen query picks its keys (compression spans, block scores, window, index scores). The readout gives the keys read per query for the toy and for $T=32\mathrm{k}$ and $128\mathrm{k}$ with the paper values. Inputs are synthetic (fixed seed, four topic directions in short runs). Indexer vectors are rounded to E4M3 with one scale per vector.
- `MLAAttentionModes` draws the MHA and MQA modes of MLA as to-scale vectors with DeepSeek-V3 sizes, with a per-token cost table.
- `content/thoughts/DS32.md`: `DSA.webp`, `mqa-mha-mode.webp` and `nsa-infllm-dsa-dinstinction.webp` replaced by the figures; `indexer-dsa-hf.webp` replaced by the `Indexer.forward` source from the DeepSeek-V3.2-Exp `inference/model.py` (fetched and compared with the screenshot).
- `content/lectures/430/notes.md`: `native-sparse-attention.webp` replaced by `SparseAttentionMask`. `MLA-kernel-Sched.svg` left as is.

Sources for the parameters: NSA section 4 ($l=32$, $d=16$, $l'=64$, $n=16$ with 1 initial and 2 local blocks, $w=512$) and Eq. 7 to 12; DeepSeek-V3.2-Exp `ModelArgs` (`index_n_heads=64`, `index_head_dim=128`, `index_topk=2048`).

## Replay

```sh
node docs/verification/sparse-attention/replay.mjs
```

It reads the embed props and captions from `content/thoughts/DS32.md`, renders both figures with `docs/verification/figure-harness.mjs`, and writes `artifacts/<figure>/` (`page.html`, `results.json`, light, dark and 320px screenshots, one screenshot per mode).

## Results

- SparseAttentionMask: 669/669 checks. All 5 patterns by click, the query slider by End, Home, PageDown and arrow keys and by value, plus a sweep of all 64 queries in every pattern. For each state: one lit row at $t$ with its frame, the lit cells equal the drawn blocks, window or top-k bars, the readout counts equal the drawn counts, the kept free blocks have the highest scores, the top-k bars are the highest index scores, $N_t$ is the branch sum. Radio arrow keys, visible focus on segments and slider, two independent instances, cleanup and a second `nav` event.
- MLAAttentionModes: 55/55 checks. Click and arrow-key mode switches. Each segment width matches its dims within 1.5px; the table matches the drawn vectors (score and value widths, FLOPs per key, cached bytes, K and V per token).
- No page errors; no horizontal scroll at 320px.

## Not checked

- The running dev watcher and the full Quartz page (no watcher in this environment); zoom at 200% and 400% was checked only through enlarged screenshots.
- Fill transitions: the state checks run with reduced motion, so computed fills are final. The screenshots run with motion on.
