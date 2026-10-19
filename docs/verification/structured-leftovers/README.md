# Structured-output figures: boundary distributions, scheduler bitmask, logit processor stack

Task: replace the last three structured-output images with figures that draw the mechanism.

## Change

- `TokenBoundaryShift` (`quartz/components/mdx/TokenBoundaryShift.tsx`) replaces `thoughts/images/vllm/json-difference-in-sampling-distribution.webp` in `thoughts/structured outputs.md`. The image sat in a collapsed callout, so the figure now follows the callout. The compressed FSM of the note forces `{"name":"` to state 9. The figure feeds that text under the natural split (`{"`, `name`, `":"`), the other seven segmentations of `name` and a one-token-per-character split. It shows the next-token distribution at state 9 over the six tokens the grammar allows (`P`, `Pa`, `Paul`, `J`, `Jo`, `John`), renormalized after the mask, with KL, TV, the mass the mask keeps and the Paul-branch mass.
- `GrammarBitmaskBatch` (`quartz/components/mdx/GrammarBitmaskBatch.tsx`) replaces `thoughts/images/constrained-proposal-scheduler.webp`. It draws one engine step as vLLM v0.8.0 implements the proposal. The requests are r0 (no grammar), r1 to r3 at grammar states 0, 9 and 22, and r4 in `WAITING_FOR_FSM`. The scheduler fills one int32 row of 4,740 words per structured request (`structured_output/__init__.py:140`). The figure draws each row as 119 cells of 40 words. `SchedulerOutput` carries the bitmask to every worker through the shared-memory broadcast (`multiproc_executor.py:111`). Every worker gathers the full logits (all-gather under V1) and applies the rows with `apply_token_bitmask_inplace` (`gpu_model_runner.py:868`, `:1004`), then samples. Rank 0 returns the output, and `accept_tokens` advances each grammar (`scheduler.py:621`).
- `LogitsProcessorTrace` (`quartz/components/mdx/LogitsProcessorTrace.tsx`) replaces `thoughts/images/vllm/pre-optimized-logit-processor-handling.webp` in both notes. In `posts/structured decoding.md` the image sat in footnote `[^sampling]`, so the figure now follows the list of limitations, and the footnote keeps the PR #5329 citation. The lanes are the classes of the original diagram plus `JSONLogitsProcessor` and `Sampler`. The calls come from the vLLM v0.5.0 source, with file and line. The batch has 4 rows; each guided row adds one per-row `logits_processor(past_tokens_ids, logits[row])` call between `compute_logits` and `model.sample`.
- Data: `content/thoughts/scripts/jump_forward_boundaries.py` writes `quartz/components/mdx/jump-forward-boundaries.json` (Qwen/Qwen3-0.6B-Base, float32 on CPU, prompt `Return a JSON object with a person's name (Paul or John) and age (20 or 30).\n`). The script also writes the allowed sets and probabilities at states 0, 9 and 22, which `GrammarBitmaskBatch` reads. The committed JSON is formatted with `oxfmt`.

The image files stay in the repository.

## Replay

```sh
# data (optional; needs torch and transformers, about 1 minute on 4 CPUs)
uv run --with torch --with transformers content/thoughts/scripts/jump_forward_boundaries.py --out quartz/components/mdx/jump-forward-boundaries.json
node_modules/.bin/oxfmt quartz/components/mdx/jump-forward-boundaries.json

# figures; VLLM_SRC is an unpacked vllm-0.5.0.tar.gz from PyPI (optional)
VLLM_SRC=/path/to/vllm-0.5.0 node docs/verification/structured-leftovers/replay.mjs
```

The replay reads the embeds from the notes with remark-mdx (`parse-captions.mjs`), renders them through `docs/verification/figure-harness.mjs` and drives Chromium. Output: `artifacts/<component>/page.html`, `results.json` and screenshots.

## Results (2026-10-19)

196 of 196 checks passed (TokenBoundaryShift 73, GrammarBitmaskBatch 93, LogitsProcessorTrace 30).

- TokenBoundaryShift: for each of the 9 splits, the drawn tape spells the split, and the salmon bar heights equal the renormalized probabilities. The table and every readout equal an oracle recomputed in the replay from the JSON. KL and TV agree with the values from the Python script. Every allowed token keeps the output inside the note's four-string language. The natural split is the tokenizer's own. Paul-branch mass is 21.2% for the natural split and ranges from 20.3% (`n·ame`) to 40.4% (one token per character), as the caption states. Arrow keys move the selection, and the focused segment shows an outline.
- GrammarBitmaskBatch: the replay checks all 12 request and worker-count pairs. For each pair, it checks N boxes and one lit logits row per worker. The bus spans every drop, and each drop ends on a box. For r1 to r3, the lit cells equal the words of the allowed token IDs and are salmon. The word count, unmasked argmax, greedy pick after the mask and the id/word/bit table equal the data. r0 has no row; r4 stays outside the batch. A second instance keeps its own state, and arrow keys work on both groups.
- LogitsProcessorTrace: the replay checks guided counts 0, 1, 2 and 4. The trace has 18 fixed calls plus one per guided row, between the logits and `sample`, and the sampler follows `sample`. With `VLLM_SRC` set, every call's name appears on its cited v0.5.0 line. At each slider position (fill), one call is lit salmon, its wires end on lane nodes, and the readout and `aria-valuetext` equal the call list. Switching to fewer rows clamps the slider. Home, ArrowRight and End walk the calls, the focused slider shows an outline, and the two instances are independent.
- Every page: light, dark and 320px screenshots, no horizontal scroll at 320px, no page errors.

Example results:

| boundary | P(Paul branch) | KL (nats) | TV    | mass the mask keeps |
| -------- | -------------- | --------- | ----- | ------------------- |
| name     | 21.2%          | 0         | 0     | 89.0%               |
| n·a·m·e  | 32.9%          | 0.0354    | 0.119 | 83.1%               |
| na·me    | 39.8%          | 0.0834    | 0.189 | 82.9%               |
| n·ame    | 20.3%          | 0.0003    | 0.010 | 87.9%               |
| chars    | 40.4%          | 0.0869    | 0.195 | 84.7%               |

## Not checked

- The rendered note pages through the Quartz build and `Zoomable` (no dev watcher in this session).
- 200% and 400% browser zoom. At 320px the three SVGs scale to about 45 to 50%, so their labels render near 4 to 5px; `Zoomable` is the reading path at that width.
- The commit `1ea291a` that the note links for the row-by-row loop. The trace follows the v0.5.0 release (2024-06-11, the PR #5329 period). GitHub source access was not available in this session.
- That the allowed sets are complete. The replay checks that every listed token is a viable continuation; completeness rests on the script's full-vocabulary scan.
- The probabilities are one prompt on one small base model. They show that the boundary changes the distribution; they are not an estimate for other models or prompts.
