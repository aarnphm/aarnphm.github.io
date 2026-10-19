# head-spectrum

## Change

`content/lectures/412/notes.md`, callout "spectral diagnostics": 14 matplotlib images (six heads, spectrum plus heatmap each, and the comma-prompt pair for layer 12 head 8) are replaced by one `HeadSpectrum` figure after the callout. The figure draws real data from `quartz/components/mdx/qwen3-head-spectra.json`.

## Data

`content/lectures/412/export_head_spectra.py` reuses the functions in `latent_projection.py` (`infer_templates`, `resolve_head_counts`, `compute_w_ov`, `latent_error`, `compute_effective_rank`). It loads `Qwen/Qwen3-0.6B` in float32 with eager attention on CPU and writes, per head: all 1024 singular values (5 significant digits), the float64 cumulative energy up to the first index where it rounds to 1 at six decimals (k = 128 for every head), `r_eff`, rank, shape, value head and the attention matrix of the head for both prompts (4 decimals).

Command (isolated env: `uv venv`, `uv pip install torch --index-url https://download.pytorch.org/whl/cpu transformers safetensors matplotlib`):

```
python content/lectures/412/export_head_spectra.py --out quartz/components/mdx/qwen3-head-spectra.json
```

A second run wrote a byte-identical file (`artifacts/export/`).

Agreement with the source images and logs:

- layer 12 head 8: `r_eff = 83.35`, as in the title of `qwen3-0.6b_layer12_head8_absurdity_spectrum.webp`; top singular values 2.312, 1.209, 1.074, 1.059 and energies 0.112, 0.143, 0.167 match the log in `tools.md`.
- layers 3 heads 3, 6, 8: top singular values match the `tools.md` logs (1.278 vs 1.279 for layer 3 head 6, σ₂ = 1.2785 rounds either way).
- every head has rank 128 = d_head, so the spectrum falls by about 6 decades after index 128, as in all six images; the tail near 1e-7 is float32 round-off.
- attention: layer 12 head 8 row "life" puts 0.63 on "The"; the comma prompt has 11 tokens with "," and row "suffering" puts 0.36 on "The", as in the two heatmaps.
- the six non-absurdity heatmaps were made with the prompt without the comma (the `tools.md` commands); the absurdity pair used the comma prompt (the note's text). The spectrum does not depend on the prompt, so the figure has a prompt toggle for the heatmap only.

## Replay

```
node docs/verification/head-spectrum/replay.mjs
```

Needs `docs/verification/figure-harness.mjs` and the shared `node_modules`. It renders the note's embed (caption read from `notes.md`) and a second instance (layer 3 head 3, no comma, k = 64), then:

- clicks every head segment and both prompt segments, and moves them with arrow keys;
- moves k with ArrowRight, Home, End and a pointer drag, and checks the thumb centre against the drawn marker in the wide and the 320px layout;
- after each step checks k, σ_k, E_k, σ_1, r_eff readouts, both marker dots, the sage band and area, the active curves, the r_eff rule and all heatmap cell weights against the JSON, with plot scales recomputed in the test;
- hovers a cell and walks the grid with arrow keys, Home and End, and checks the readout text, the salmon mark position, the focus frame and the causal-mask stop;
- checks ramp direction in both themes, instance independence, unique radio names, reduced motion, light, dark and 320px screenshots, horizontal scroll and page errors.

Result: 48/48 checks passed (`artifacts/results.json`).

## Not checked

- The real Quartz build of the note (no dev watcher in this environment); the harness uses the same esbuild, Sass and inline-script loaders.
- Safari and Firefox range-thumb geometry; alignment is checked in Chromium only.
- SPA navigation cleanup across pages.
