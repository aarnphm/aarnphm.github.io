# Graph traversal, block diagram and radix tree figures

Task: replace eight note images with three figures that draw the mechanism.

## Change

- `GraphTraversal` (`quartz/components/mdx/GraphTraversal.tsx`) replaces `images/example-graph-dfs.webp` in `sfwr-2c03/Graphs.md`. It draws the course graph on $n_0,\ldots,n_9$ with the edges of the image. The cycle $n_0, n_7, n_2, n_3$ is drawn as a square. Pure functions run DFS-R and BFS from the note, with neighbours in ascending index order, from every source. They record each entry, return and neighbour scan. The figure shows the call stack or queue, $d$/$f$ times (DFS) or order, distance and parent (BFS), and the tree edges. The server sends all 20 runs; `graph-traversal.inline.ts` only applies a precomputed step.
- `BlockDiagram` (`quartz/components/mdx/BlockDiagram.tsx`) replaces seven images. Each configuration is data: a forward chain of sums, blocks and samplers, an optional feedback chain, and side inputs. A pure reducer applies the rules:
  - series: adjacent continuous blocks multiply;
  - sampler: a block that a sample sequence drives gives a pulse transfer function, written $G(z)$ for one named factor and $\mathcal{P}_T\{\cdot\}$ for a product;
  - series of sequences: $z$-blocks multiply, output side first;
  - parallel: two branches from one takeoff point that meet at a junction add, $G_1 \pm G_2$. A plain gain plus $N/D$ is also written over $D$ and with the gain factored out, $K_1 + K_2/s = (K_1s + K_2)/s = K_1(s + K_2/K_1)/s$, and the readout gives its zero and pole;
  - feedback: $G/(1 \mp GH)$ for the continuous loop. An empty feedback chain is unity feedback. When a factor is a fraction, the result is also written with the inner fractions cleared, $N_fD_h/(D_fD_h \mp N_fN_h)$;
  - moving a block past a summing junction, checked by expanding the output of both diagrams;
  - sampled single loops: the samplers cut the loop into continuous segments. Each segment gives one relation. The junction segment gives the continuous relation and then its sampled form. Solving gives the result, or no $C(z)/R(z)$ when the reference stays inside $\mathcal{P}_T$.

  The server renders every step (drawing, rule, equation, readout). `block-diagram.inline.ts` only shows the chosen variant and step.

- `PatriciaTrie` (`quartz/components/mdx/PatriciaTrie.tsx`) replaces `thoughts/images/Patricia_trie.svg` in `Radix tree.md`. A pure insert function returns a new tree for each key. It splits an edge where the key diverges. The keys are those in the SVG (romane, romanus, romulus, rubens, ruber, rubicon, rubicundus). The readout compares radix nodes with uncompressed trie nodes. `RadixPrefixTree` draws a fixed token-prefix cache scenario, so a separate component was simpler than extending it.

Embeds:

| Note                              | Old image                                    | Figure                                     |
| --------------------------------- | -------------------------------------------- | ------------------------------------------ |
| `sfwr-2c03/Graphs.md`             | `example-graph-dfs.webp`                     | `GraphTraversal mode="dfs" start={0}`      |
| `sfwr-3dx4/Block Diagrams.md`     | `feedback-loop-transfer-function.webp`       | `BlockDiagram config="feedback"`           |
| `sfwr-3dx4/Block Diagrams.md`     | `summing junction.webp`                      | `BlockDiagram config="summing-junction"`   |
| `sfwr-4aa4/closed loop system.md` | `closed-loop-z-block-diagram-reduction.webp` | `BlockDiagram config="sampler-cascade"`    |
| `sfwr-4aa4/closed loop system.md` | `model-open-loop-system.webp`                | `BlockDiagram config="open-loop"`          |
| `sfwr-4aa4/closed loop system.md` | `closed-loop-sampled-data-system.webp`       | `BlockDiagram config="sampled-feedback"`   |
| `sfwr-4aa4/closed loop system.md` | `closed-loop-tf-sensing-device.webp`         | `BlockDiagram config="sensing-device"`     |
| `sfwr-4aa4/closed loop system.md` | `closed-loop-tf-digital-controller.webp`     | `BlockDiagram config="digital-controller"` |
| `sfwr-3dx4/Root locus control.md` | `idea-integral-compensator-impl.webp`        | `BlockDiagram config="pi-controller"`      |
| `sfwr-4aa4/PID controller.md`     | `prop-control.webp`                          | `BlockDiagram config="proportional"`       |
| `Radix tree.md`                   | `Patricia_trie.svg`                          | `PatriciaTrie`                             |

The image files stay in the repository.

## Replay

```sh
node docs/verification/graphs-blocks-trie/replay.mjs
```

It uses `docs/verification/figure-harness.mjs` (Quartz esbuild, Sass and inline-script loaders, then Chromium). `CHROMIUM` overrides the browser path. Output: `artifacts/<component>/page.html`, `results.json` and screenshots.

## Results (2026-10-19)

162 of 162 checks passed (GraphTraversal 42, BlockDiagram 86, PatriciaTrie 34). The BlockDiagram run now has nine instances; the `pi-controller` and `proportional` configurations were added later (task `blocks-relalg`), and the whole replay was run again with them.

- GraphTraversal: all 27 DFS steps from $n_0$, stepped with the next button. At each step the open vertices are the call stack, the salmon vertex is its top, and shown $d$/$f$ cells match the vertex states. The final times match a hand trace: $n_0$ 1/16, $n_3$ 2/7, $n_2$ 3/6, $n_7$ 4/5, $n_4$ 8/9, $n_9$ 10/15, $n_5$ 11/14, $n_6$ 12/13. The BFS distances and parents from $n_0$ match too. For all 20 mode and source pairs, the marked count is 8 (2 from $n_1$ or $n_8$), the tree has one edge less, and the readout agrees. The checks also cover the slider and radio keyboard, the second instance, focus outline and cleanup.
- BlockDiagram: for every variant and step, one panel is shown and each rule step lights its parts. Final results equal the notes:
  - feedback: $H_1/(1+H_1H_2)$, positive $H_1/(1-H_1H_2)$;
  - cascade: $G_2(z)G_1(z)$ with the sampler, $\mathcal{P}_T\{G_1(s)G_2(s)\}$ without;
  - open loop: $C(z)=G(z)D(z)E(z)$ with $G(s)=\frac{1-e^{-sT}}{s}G_p(s)$ and $G(z)=(1-z^{-1})\mathcal{P}_T\{G_p(s)/s\}$;
  - sampled error: $G(z)/(1+\mathcal{P}_T\{G(s)H(s)\})$;
  - sensing device: $C(z)=\mathcal{P}_T\{GR\}/(1+\mathcal{P}_T\{GH\})$ and no $C(z)/R(z)$;
  - digital controller: $G_1(z)G_2(z)/(1+G_1(z)\mathcal{P}_T\{G_2(s)H(s)\})$;
  - PI controller (Nise Figure 9.8): steps diagram, parallel, loop, feedback. The parallel step derives the note's $G_c(s) = K_p + K_i/s = K_p(s+K_i/K_p)/s$ with $K_1 = K_p$, $K_2 = K_i$; the replay reads that equation from the note. The readout gives zero $-K_2/K_1$ and pole $0$. The result is $K_1(s+K_2/K_1)G(s)/(s + K_1(s+K_2/K_1)G(s))$. The diagram draws three blocks and two junctions; the parallel step leaves two blocks and one junction;
  - proportional: $E(s)=R(s)-Y(s)$, $Y(s)=K_pG_p(s)E(s)$ and $K_pG_p(s)/(1+K_pG_p(s))$, which the replay finds in the PID note.

  Every KaTeX label of every step fits its box.

  The intermediate relations of the notes appear in the steps. Both summing-junction moves give the same expanded output. Seven instances on one page have unique ids.

- PatriciaTrie: at each of the 8 states, every key node spells its key from the root, and the drawn nodes equal the readout. Trie nodes equal the distinct prefixes, and edge symbols equal trie nodes minus one. Each insert lights one new edge. The final edge labels equal the SVG's: r, om, an, e, us, ulus, ub, e, ns, r, ic, on, undus. The final tree has 14 nodes; the uncompressed trie has 28. A second instance with the note's `car`, `cart`, `cat` gives edges ca, r, t, t, with `car` terminal above `cart`.
- Every page: light, dark and 320px screenshots, no horizontal scroll at 320px, no page errors.

## Not checked

- The rendered note pages through the Quartz build and `Zoomable` (no dev watcher in this session).
- 200% and 400% browser zoom.
- At 320px the graph and tree SVGs scale to about 55%, so their 10px labels render near 6px. `Zoomable` is the reading path at that width.
- The parallel rule factors only a plain gain plus one fraction; other sums stay as $G_1 \pm G_2$. No configuration uses them.
- The `content-visibility: auto` frame needs a frame to render after a long scroll before a click can hit it, so the replay waits 150ms before the keyboard checks.
- The source of "Figure 5.7" in the summing-junction image; the course textbook in the repository is a Git LFS pointer.
