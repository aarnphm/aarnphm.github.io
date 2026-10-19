# ml-small: small ML figures

## Change

Four figures replace six image embeds.

| Note                                        | Image                                                           | Figure                             |
| ------------------------------------------- | --------------------------------------------------------------- | ---------------------------------- |
| `thoughts/sparse autoencoder.md`            | `gated-sae-architecture.webp`                                   | `GatedSAE view="gated"`            |
| `thoughts/sparse autoencoder.md`            | `gated_jump_relu.webp`                                          | `GatedSAE view="jumprelu"`         |
| `thoughts/optimization.md`                  | `convergence-vs-step-side-momentum.webp`                        | `HeavyBallRate`                    |
| `sfwr-4ml3/Convolutional Neural Network.md` | `sparse-connectivity-images.webp`, `exploiting-invariance.webp` | `ConvLocality view="connectivity"` |
| `sfwr-4ml3/Convolutional Neural Network.md` | `max-pooling.webp`                                              | `ConvLocality view="pooling"`      |
| `thoughts/FFN.md`                           | `residual-network.webp`                                         | `ResidualBlock` (static)           |

The image files stay on disk.

## Replay

```sh
node docs/verification/ml-small/replay.mjs
```

It needs `node_modules` at the worktree root and Chromium at `/opt/pw-browsers/chromium` (or `CHROMIUM`). It uses `docs/verification/figure-harness.mjs`. Output goes to `artifacts/<figure>/` (`page.html`, `results.json`, light, dark and 320px screenshots, and state screenshots).

## Results

40 of 40 checks pass:

- GatedSAE, 11 checks. At $\theta=0.35$, $u$, $f_\text{mag}$ and $\tilde f$ equal the paper figure's vectors. For all 51 thresholds, the drawn cells obey $\pi_\text{mag}=e^{r}\odot u+b_\text{mag}$, $\pi_\text{gate}=u+b_\text{gate}$, $f_\text{gate}=\mathbb 1[\pi_\text{gate}>0]$ and $\tilde f=f_\text{gate}f_\text{mag}=\operatorname{JumpReLU}_\theta(\pi_\text{mag})$. On the plot, active points lie on the drawn identity line and inactive points on the zero axis. The keyboard check passes.
- HeavyBallRate, 8 checks. In 45 $(\alpha\lambda,\beta)$ cases, the rate readout equals the spectral radius from Gelfand's formula on the 2×2 iteration matrix. The marker, guide, sage band edge $2(1+\beta)$ and iterates match the drawn tick axes. The settle count matches a separate simulation. At $\beta=0$ the curve is the V. The keyboard checks pass for both sliders.
- ConvLocality, 12 checks. The pooling view checks every window for each $F\in\{2,3\}$, $S\in\{1,2\}$: the window cells, the max cell, the output value and the skipped-input count. $F=2,S=2$ gives 6 8 / 3 4. The connectivity view checks every unit in every mode: patch cells, the largest weight id against the weight-count readout (576, 144, 9), shared ids that are the same for all units, and dense and local ids that are distinct. Keyboard and focus checks pass.
- ResidualBlock, 5 checks. Every edge ends on a node or on the sum circle.
- Notes, 4 checks. Each new fence parses with remark-mdx and has a caption.

## Not checked

- The full Quartz build and the dev watcher (none runs in this environment). The note fences are parsed with the same remark-mdx parser, but they are not rendered through `codeblock.ts`.
- 200% and 400% browser zoom. The SVGs scale with the viewBox. At 320px the gated diagram scrolls inside its frame.
- SPA navigation cleanup across pages.
