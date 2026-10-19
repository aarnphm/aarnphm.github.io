# KV-Compress and all-reduce figures

Task: replace three images with interactive figures.

- `content/thoughts/KV compression.md`, KV-Compress section: `kv-compress-vllm.webp` and `paged-attention-block-kv-compress.webp` become two embeds of `KVCompressBlocks` (`view="vllm"` and `view="evict"`).
- `content/thoughts/tsfm/tangents-3.md`, all-reduce section: `all-reduce-meta.webp` becomes `AllReduceSteps`.

The image files stay in the repository. `comparison-ddp-fsdp.webp` in `tangents-3.md` is not touched.

## Change

- `quartz/components/mdx/KVCompressBlocks.tsx`. `PagedKVTable` was not extended: it draws one fixed page table with append and evict actions and has no per-head blocks, block metrics or ordered steps. The new component has three tabs:
  - Eviction: a pure function `evictBlocks` runs the paper's ten steps on the example read from its figure (two KV heads, block size 2, evict 2 blocks; head 0 holds 0.4, 2.5, 0.6, 1.7, 0.1 and an empty slot at logical 0 to 5, head 1 holds 1.2, 2.3). The empty slot counts as metric 0 and a block's metric is its largest entry, as the paper states. The function returns all ten frames and the readout. SSR draws every frame, and the inline script only switches the visible one.
  - vLLM blocks and KV-Compress blocks: the same per-head KV counts in both layouts ($l = 2$, $H = 2$, block size 2). Layer 0 uses the example's head lengths (5, 2), and layer 1 (3, 4) is illustrative. Block counts, block tables, context lengths and slot totals are computed from those lengths.
- `quartz/components/mdx/AllReduceSteps.tsx`. `RingRotation` (ring attention KV rotation) and `TreeReduction` (flash-decoding merge tree) do not draw reduce-scatter or all-gather. `ringAllReduce(n)` simulates ring reduce-scatter (GPU $i$ sends chunk $(i - t) \bmod N$ to $i + 1$, which adds it) and all-gather (forwards full chunks). It runs for $N \in \{2, 3, 4, 6\}$, with $2(N - 1)$ steps each. The readout shows the sends of the step, $S/N$ per send, the bytes sent so far and $2(N-1)/N \cdot S$.
- Controls: native tabs with arrow, Home and End keys; native radio groups for step and $N$; native prev, next and reset buttons. When a button becomes disabled at a bound, focus moves to its opposite button.

## Replay

There is no `quartz/scripts/dev.ts` watcher in the cloud session, and `pnpm install` fails behind the proxy. The replay uses `docs/verification/figure-harness.mjs`, which uses the Quartz esbuild, Sass and inline-script loaders inside `Zoomable`. It reads the props of each embed from the notes.

```sh
node docs/verification/kv-compress-collectives/replay.mjs
```

Output: `artifacts/kv-compress-blocks/` and `artifacts/all-reduce-steps/` (`page.html`, `results.json`, screenshots).

## Results (2026-10-19)

KVCompressBlocks: 72 of 72 checks passed.

- For each of the ten steps, the drawn cells (metric, logical index, mark, block-metric outline) are equal to a table that was read independently from the paper figure. Exactly one frame and one step text are visible. The eviction range shows only in steps 7 to 9, and the move arc only in step 8.
- Readout: h0 evicts 2 blocks and h1 evicts 0. The evicted KVs are 0.1, 0.4 and 0.6. One KV moves (1.7 from index 3 to 0). The freed blocks are p1 and p7. Each head keeps 2 KVs.
- Layout: vLLM has 1 block-table row, and its lit block id covers 8 slots ($2 \times 2 \times 2$). KV-Compress has 4 rows (L0 h0 to L1 h1), its lit block is 2 slots, and it allocates 16 slots against 24 for vLLM, for 14 KVs.
- Tabs: click, ArrowRight wraparound, Home, End, roving tabindex and visible focus all work. The second instance keeps its own state. After cleanup, the listeners are removed.

AllReduceSteps: 126 of 126 checks passed.

- For every $N$ and every step, the cell labels are equal to an independent ring simulation in the replay. Exactly $N$ senders and $N$ receivers are lit in each step after step 0. After $N - 1$ steps of reduce-scatter, GPU $i$ holds $\Sigma$ of chunk $i$. After $2(N-1)$ steps, all cells hold $\Sigma$.
- Sent so far is $t/N \cdot S$ for each step $t$. The total readout is equal to $2(N-1)/N$.
- The radio arrow keys, Enter and Space on next, and visible focus work. Focus moves to prev when next becomes disabled. After cleanup, the click listener is removed.
- At 320px, there is no horizontal page scroll. There are no page errors.

One run exited with a Node error before it printed its checks (the trace was not kept). The four runs after it passed with no change to the components.

## Not checked

- The rendered note pages through the full Quartz build.
- 200% and 400% browser zoom.
- At 320px, the eviction drawing and the $N = 6$ grid scale to about 60%, so their labels render near 6px. `Zoomable` is the reading path at that width.
