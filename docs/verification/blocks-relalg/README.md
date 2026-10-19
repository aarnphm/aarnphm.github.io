# PI and proportional block diagrams, relational algebra expression tree

Task `blocks-relalg`: replace three note images.

## Change

- `BlockDiagram` (`quartz/components/mdx/BlockDiagram.tsx`) gets a parallel rule, unity feedback and two configurations. The replay and its results are in `docs/verification/graphs-blocks-trie/README.md`.
  - `pi-controller` replaces `images/idea-integral-compensator-impl.webp` in `sfwr-3dx4/Root locus control.md` (Nise Figure 9.8). The steps are diagram, parallel ($K_1 + K_2/s = K_1(s+K_2/K_1)/s$, with zero and pole), loop and feedback.
  - `proportional` replaces `prop-control.webp` in `sfwr-4aa4/PID controller.md`. The figure uses the note's notation ($R$, $E$, $U$, $Y$, $G_p$). The course image draws the plant on the return path and names the set point $u(t)$; the caption says so.
- `RelAlgTree` (`quartz/components/mdx/RelAlgTree.tsx`, `styles/relAlgTree.scss`, `scripts/rel-alg-tree.inline.ts`) replaces `expression-tree-relalg.webp` in `sfwr-3db3/Relational Algebra.md`. It draws the tree $\pi_\text{name}(\sigma_{\text{addr}=\text{“Maple St.”}}(\text{Bars})) \cup \rho_{R(\text{name})}(\pi_\text{bar}(\sigma_{\text{price}<3 \text{ AND } \text{beer}=\text{“Bud”}}(\text{Sells})))$ with KaTeX labels. A pure evaluator (set semantics) runs it over the note's Sells and Bars rows. Each node is a radio; selecting one shows its inputs, per-tuple test results for a selection, and its result. The server renders all eight panels; the script only applies the chosen one. Below a 460px container, an indented outline with its own radio group replaces the tree; the script keeps both groups on the same node.
  - Bars: the note's table has columns (bar, addr) and says Bars.name became Bars.bar for the join example. The query uses Bars(name, addr), so the figure keeps the note's rows and names the column name. The caption states this.

The image files stay in the repository.

## Replay

```sh
node docs/verification/blocks-relalg/replay.mjs
node docs/verification/graphs-blocks-trie/replay.mjs
```

The first renders two `RelAlgTree` instances (the first with the note's caption, the second with `select="sigma-bud"`) through `docs/verification/figure-harness.mjs` and writes `artifacts/rel-alg-tree/` (page, screenshots, `results.json`).

## Results (2026-10-19)

- `RelAlgTree`: 56 of 56 checks passed. The replay parses the Sells and Bars Markdown tables from the note and checks that the figure's stored relations equal them. It works the query by hand over those rows (answer Joe’s, Sue’s). For each of the eight nodes, clicked in turn, it checks: one panel; the salmon node; the sage subtree; dark edges only inside the subtree; the result equals the hand-worked relation; each input table equals its child's result; for selections, the sage rows are exactly the rows whose tests are all true, and they form the result. Also checked: ArrowRight walks the nodes bottom-up, focus outline, second instance independent, unique radio names, outline at 320px (click and keyboard, synced to the wide group), labels fit their boxes, no 320px page scroll, no page errors, cleanup.
- `graphs-blocks-trie` replay: 162 of 162 (GraphTraversal 42, BlockDiagram 86, PatriciaTrie 34).

## Not checked

- The note pages through the Quartz build and `Zoomable` (no dev watcher in this session).
- 200% and 400% browser zoom.
- At 320px the display expression above each panel scrolls sideways inside its box.
