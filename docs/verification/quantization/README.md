# Quantization figures

Task: replace the two image embeds in `content/thoughts/quantization.md` with computed figures, and add one figure for the MX block conversion.

## Change

- `quartz/util/minifloat.ts` is the shared model. It rounds a double to FP32, BF16, FP16, FP8 E5M2 and E4M3, FP6 E3M2 and E2M3, and FP4 E2M1, to nearest with ties to even. It gives the field bits, the range (largest finite, smallest normal, smallest subnormal) and the level set of each format. It also holds the MX block conversion of Rouhani et al. (2023), Algorithm 1: shared exponent $\lfloor\log_2\max_i|V_i|\rfloor - e_{\max}$, then element rounding with clamping.
- `NumberFormatBits` replaces `quantisation-format.webp` (the Baseten figure). Seven rows (FP32, BF16, FP16, FP8 E5M2, FP8 E4M3, FP4 E2M1, INT8) share one 32-column grid. Six preset inputs. The selected row's bits are decoded with the note's formula, with absolute and relative error and the format range. IEEE rows overflow to infinity; FP8 and FP4 saturate. INT8 uses the note's symmetric example, $S = 1/127$.
- `MXBlockScale` is new, after the conversion recipe in the `mxfp4` section. One block of 32 seeded standard normal values, optionally with element 11 set to 7 (the note's example) or 24. Four element formats (MXFP4 E2M1, MXFP6 E2M3 and E3M2, MXFP8 E4M3). It draws inputs, dequantized values, the level ticks $SP$ and per-element error.
- `MXComputeFlow` replaces `compute-flow-mxformat.webp` (Figure 2 of the MX paper). It draws the three products of a training step ($A_{i-1}W_i$, $A_{i-1}^\top E_i$, $E_iW_i^\top$) element by element, with $M = 32$, $K = 64$ and $N = 32$, and blocks of 32 along the reduction axis. A table counts where the two MXFP4 copies of each tensor differ, because quantization and transposition do not commute.
- The image files stay in the repository.

## Replay

There is no `quartz/scripts/dev.ts` watcher in this cloud session. The replay renders each component through the shared harness `docs/verification/figure-harness.mjs` (the Quartz esbuild, Sass and inline-script loaders) and drives it in Chromium:

```sh
node docs/verification/quantization/replay.mjs
```

Output: `artifacts/<figure>/page.html`, `results.json` and light, dark, 320px and keyboard-focus screenshots. Each page holds two instances.

## Results (2026-10-19)

182 of 182 checks passed (104 + 49 + 29).

- NumberFormatBits: for all 6 inputs × 7 formats, the stored bits match known code points. FP32 is checked against `DataView.setFloat32`. BF16 and FP16 are checked against IEEE tables, for example $\pi$ → `0x4049` and `0x4248`, and $10^{-5}$ → FP16 subnormal `0x00A8`. FP8 and FP4 are checked against OCP code points, for example $\pi$ → E4M3 `0x45` (3.25) and E5M2 `0x42` (3), $1000$ → E4M3 `0x7E` (448, saturated), $70\,000$ → E5M2 `0x7B` (57344) and FP16 `0x7C00` (infinity). INT8 is checked against the note's example $0.3$ → `0x26` (38). For each case, the drawn bit cells equal the stored bits. Range readouts equal the formats' published limits (65504, 448, 57344, 6). Arrow keys move inside both radio groups.
- MXBlockScale: for all 12 states, the flushed-to-zero count equals the drawn bars, every dequantized bar ends on a drawn level tick, and one block maximum is lit. For maximum 7 in E2M1: $S = 1$, one value clamped, and the bar is $6/7$ of the input. For outlier 24: shared exponent $4-2=2$ (E8M0 code 129) in E2M1, and $4-8=-4$ in E4M3. Bits per element are 4.25 and 8.25.
- MXComputeFlow: for each product, only its panel shows. The operands have the expected row and column counts and $R/32$ blocks of 32 elements along the reduction. One row and one column are lit, and the two highlighted table rows are the copies that this product uses. The copy pairs differ in 782/2048 ($A$), 241/2048 ($W$) and 490/1024 ($E$) values.
- All three: the second instance keeps its own state, cleanup removes the listener, there is no horizontal page scroll at 320px, and there are no page errors.

## Not checked

- The rendered note page through the Quartz build. The jsx fences were not run through the site's MDX pipeline.
- 200% and 400% browser zoom.
- At 320px, the MXBlockScale and MXComputeFlow SVGs scale to about 45%, so their 9 to 10px labels render near 4 to 5px. `Zoomable` is the reading path at that width.
- The checks switch transitions off before they read computed colours. The eased transitions themselves were not inspected.
