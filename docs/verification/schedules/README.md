# Schedule and priority-protocol figures

Task: replace the schedule images in `sfwr-3db3/Transaction.md` and the PIP/PCP slides in `sfwr-4aa4/PIP and PCP.md` (under `content/thoughts/university/twenty-four-twenty-five/`) with figures that draw the mechanism.

## Change

- `ScheduleClassifier` (`quartz/components/mdx/ScheduleClassifier.tsx`). A pure classifier takes a schedule of $R$, $W$, commit and abort operations. It computes the following properties from the note's definitions:
  - serial: each transaction's operations, its commit or abort included, are contiguous.
  - conflict serialisable: the precedence graph over all transactions, aborted ones included, is acyclic.
  - recoverable: a reader commits only after every writer it read from has committed.
  - ACA: every read sees committed data.
  - strict: no other transaction reads or overwrites an item that a transaction wrote before that transaction ends.

  A read sees the latest preceding write, unless that write's transaction aborted first. The figure draws a T1/T2 timeline, the precedence graph and the slide's Venn diagram. The schedule's region is the salmon fill; sage cells mark the operations that keep it out of the next inner class. It has nine presets, one per Venn region. Three are the slide tables, with commits added because the slides omit them. One is the note's lost-update example. Five are constructed for the remaining regions. `view="single"` draws one preset without the picker.

- `PriorityProtocolTimeline` (`quartz/components/mdx/PriorityProtocolTimeline.tsx`). A discrete-time, single-CPU, preemptive simulation of a task set under no protocol, PIP or PCP. Inheritance is transitive and is recomputed at every scheduling decision. PCP admits a lock only when $p_{\mathrm{eff}}(T) > C_{-T}$. Unlocks take effect at the end of the unit that finishes a critical section. A deadlock is a cycle of waiting tasks when nothing can run. Blocking counts the units in which a task is released and not running while a task of lower base priority runs. Chained blocking means two or more distinct lower-priority critical sections block one task.
  - PIP slide: values read from the image (T4: 3 run, 6 in $S$, 3 run from $t=0$; T1: 3, 3 in $S$, 3 from $t=5$; T2 and T3: 3 units each). The slide does not show T2 and T3 releases; they are assumed at $t=9$ and $t=10$, inside T1's wait. Under PIP the simulation reproduces the slide bar for bar.
  - PCP slide: the image has no time axis. Durations are read from bar lengths at about 34.5 px per unit (A: 1 run, 6 in $S_3$, 4 in $S_2 \subset S_3$, 4 in $S_3$; B from $t=4$: 2 run, 3 in $S_2$, 3 in $S_3 \subset S_2$, 3 in $S_2$; C from $t=9$: 2 run, 4 in $S_1$, 2 run). Under PCP the simulation reproduces every event on the slide: B ceiling-blocked at 6, C locks $S_1$ at 11, A locks $S_2$ at 17, B runs from 25 to 34.
  - Chained (synthetic): shaped after the chained-blocking case in Sha, Rajkumar and Lehoczky (1990).
- SSR renders every state. The inline scripts only switch `hidden` panels and root attributes.

## Replay

```sh
node docs/verification/schedules/replay.mjs
```

The replay needs repository dependencies, `playwright-core` and `docs/verification/figure-harness.mjs`. It reads the props from the notes' `jsx` fences with `remark-mdx`, renders them through the shared harness, drives every control, and writes pages, screenshots and `results.json` files to `artifacts/schedule-classifier/` and `artifacts/priority-protocol-timeline/`.

## Results (2026-10-19)

- ScheduleClassifier: 75 of 75 checks passed. For every preset:
  - the region equals a hand derivation;
  - the lit Venn cells are exactly that region's cells;
  - the readout yes/no rows agree with the region;
  - cycles draw both edges;
  - the witness cells are present.
    Each of the nine regions holds exactly one preset. The single views have no picker. The keyboard and focus checks, cleanup, the light, dark and 320px screenshots, and the no-page-scroll check also passed.
- PriorityProtocolTimeline: 104 of 104 checks passed. All nine task-set and protocol states match hand-traced blocking and completion times. Drawn bars sum to each task's run time and end at its completion time. The deadlock rule matches the readout. Salmon inherited bars appear only under a protocol. The second instance keeps its state. The keyboard, cleanup, screenshot and page-scroll checks also passed.

Hand-traced blocking / completion:

| Task set | none                       | PIP                       | PCP                       |
| -------- | -------------------------- | ------------------------- | ------------------------- |
| PIP      | T1 10/24, T2 0/12, T3 0/15 | T1 4/18, T2 3/21, T3 2/24 | same as PIP               |
| PCP      | deadlock at 20 (A, B)      | deadlock at 20 (A, B)     | C 0/17, B 11/34, A 0/25   |
| chained  | TH 7/18, chained           | TH 6/17, chained          | TH 2/13, one section only |

## Not checked

- The rendered note pages through the Quartz build. The session had no `dev.ts` watcher. The replay uses the Codeblock parser's `remark-mdx` step, not the full transformer.
- 200% and 400% browser zoom.
- At 320px, the Venn labels render near 5px, and the Gantt chart scrolls inside the frame (minimum width 30rem). `Zoomable` is the reading path at that width.
