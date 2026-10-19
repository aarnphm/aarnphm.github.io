# KV budget allocation figure

Task: replace the PyramidKV and Ada-KV paper images in `content/thoughts/KV compression.md` with one interactive figure, `KVBudgetAllocation`.

## Change

- `quartz/components/mdx/KVBudgetAllocation.tsx` holds the model and the drawing. The model makes synthetic window votes $C$ for 4 layers, 3 heads and 16 prefix tokens. Low layers spread attention, high layers concentrate it, and one head per layer is dispersed. It applies five policies at three per-head budgets ($k = 2, 4, 6$):
  - Full keeps all 192 prefix entries.
  - StreamingLLM keeps the sink token and the newest $k-1$ entries.
  - SnapKV keeps the top-$k$ entries of each head. Pooling is omitted.
  - PyramidKV gives layer $\ell$ a budget that falls linearly from $2k - k/\beta$ to $k/\beta$, $\beta = 4$, rounded by largest remainder so the total stays $Lk$.
  - Ada-SnapKV keeps $Hk$ entries per layer: the top entry of each head, then the top votes pooled across the layer's heads.
- SSR renders the initial state and serializes all 15 states on the figure root. `kv-budget-allocation.inline.ts` only applies a precomputed state, so the drawing, row values and readout come from one source.
- Controls are native radio groups (policy, budget). Arrow keys move inside a group.
- The note embeds the figure twice: PyramidKV initial state in the Pyramid-KV section, Ada-SnapKV in the Ada-KV section. The image files stay in the repository.

## Replay

The cloud session had no `quartz/scripts/dev.ts` watcher, and `pnpm install --frozen-lockfile` fails there: the proxy refuses `codeload.github.com` (the `basedpyright-*-source` tarballs). The replay uses the same esbuild, Sass and inline-script loaders as `quartz/cli/handlers.js`, but renders only this component. It is component-level evidence, not a rendered-site check.

With repository dependencies and `playwright-core` installed:

```sh
node docs/verification/kv-budget-allocation/replay.mjs
```

`CHROMIUM` overrides the browser path (default `/opt/pw-browsers/chromium`). Output goes to `artifacts/`: `page.html`, `results.json` and screenshots.

## Results (2026-10-19)

51 of 51 checks passed:

- SSR state for PyramidKV, $k = 4$: layer budgets `7 · 5 · 3 · 1`, row counts match, drawn kept cells equal the readout.
- For all 15 states, the drawn kept cells equal the readout and the policy rule shown is the selected one. Every budgeted policy spends exactly $12k$ entries.
- The second instance on the page keeps its own state.
- ArrowRight moves the policy selection. The focused segment shows a solid outline.
- At 320px, there is no horizontal page scroll.
- After cleanup, a radio change no longer updates the figure.

Readout at $k = 4$ (48 of 192 entries):

| Policy       | Layer budgets | Mean mass kept | Worst head |
| ------------ | ------------- | -------------- | ---------- |
| StreamingLLM | 4 · 4 · 4 · 4 | 24.5%          | 0.7%       |
| SnapKV       | 4 · 4 · 4 · 4 | 66.9%          | 29.8%      |
| PyramidKV    | 7 · 5 · 3 · 1 | 65.4%          | 37.7%      |
| Ada-SnapKV   | 4 · 4 · 4 · 4 | 68.3%          | 22.7%      |

On these votes, Ada-SnapKV gives the highest mean and PyramidKV the highest worst head. The numbers come from the synthetic votes, not from a model.

## Not checked

- The rendered note page through the Quartz build and `Zoomable`.
- 200% and 400% browser zoom.
- At 320px the SVG scales to about 55%, so its 9px labels render near 5px. `Zoomable` is the reading path at that width.
