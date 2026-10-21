# Assignment: performance hill climb, group `{{GROUP}}`

Workspace: `{{WORKTREE}}` (a git worktree of aarnphm's garden, branch `{{BRANCH}}`, at `{{COMMIT}}`). Work only there. The main checkout and other worktrees are other people's; never touch them. Never commit, push, stash-drop, rebase or reset; leave your changes in the working tree. Worker id: `{{WORKER}}`.

Your dev watcher is already running: it rebuilds the site on save and serves it at `{{URL}}`. Its log is `{{WORKTREE}}/.base-perf/dev.log`. After every edit, run `{{RUN}} wait-build {{NAME}}` and only then measure; it returns when a build:ready newer than your last wait is served (a failed build prints the log tail and exits 1; fix the error first). Never start, stop or restart a watcher, build or server yourself.

## Objective

Make the pages in your group faster to interact with, measured by the bench, without changing what they render or do. The bench (`{{RUN}} bench {{NAME}} <out-dir> [--reps N]`) opens each page in the real Helium browser, runs a fixed interaction script and reports, per scenario, `interaction_ms` (the primary metric: wall time of the scripted interactions), `task_ms` (main-thread time), `longtask_ms`, `loaf_block_ms` (long animation frames), `frames_dropped`, `fps_min` and `worst_frame_ms` (a rAF sampler during every step), `ready_ms` (navigation to view ready), `nodes`, `listeners`, `heap_mb`, `dom_heap_mb` (Blink heap), layouts, style recalcs and per-step times with fps. Lower is better everywhere except `fps_min`. The baseline and the final `ab` also carry Lighthouse medians (`lh_score`, `lh_tbt_ms`, `lh_lcp_ms`, ...; `{{RUN}} bench ... --lighthouse 2` adds them to any run, about 25 s per scenario per run); they are reported, not gates. Scenarios: {{SCENARIOS}}. Read `{{HARNESS}}/scenarios.py` for exactly what each step does; it is the workload and you must not edit it. The bench may print `waiting for ...lock`, `load N > M ...; waiting` or `tab hidden; keep part of the bench window on screen` and sit for some minutes: other workers share the browser and run one bench at a time, and a saturated machine is not measured until it calms (or the run is flagged `LOAD HIGH`, which makes its numbers unusable for a decision; wait and bench again). The run script already handles the sandbox (uv cache in /tmp, the shared lock opened read-only, the broker socket); run it as given, with no environment of your own, and report the exact error if it still fails.

Your baseline is already measured at `{{RUN_DIR}}/baseline/summary.md` (5 reps). Read it first and pick the scenario and step where the time goes.

## Loop

1. Form one hypothesis about where the cost is (read the view's source and stylesheet; `{{HARNESS}}/probe.py <url> --eval '<expr>'` evaluates an expression in the page, so you can count nodes, time a call, or read computed styles). Write it as a line in `{{RUN_DIR}}/experiments.md` before you edit: `N. <hypothesis> -> <files>`.
2. Edit only files you own (below). Keep the change small and specific to the hypothesis.
3. `{{RUN}} wait-build {{NAME}}`, then `{{RUN}} bench {{NAME}} {{RUN_DIR}}/cand-N --reps 3`, then `{{RUN}} compare {{RUN_DIR}}/baseline {{RUN_DIR}}/cand-N`. ACCEPT means the primary metric's median beat the baseline's best rep and no guard metric (task, long tasks, ready, dropped frames, nodes, listeners, JS heap) fell outside the baseline's range and no check broke.
4. Run the checks: `{{RUN}} check {{NAME}}` (tsgo, oxlint, oxfmt on your changed files, and the view's test file). Baseline failures listed in `{{RUN_DIR}}/baseline/checks.txt` are known; you may not add one.
5. Keep the change if ACCEPT on every scenario it touches and no other scenario got worse and the checks pass. Otherwise revert it exactly (`git checkout -- <file>` for each file you changed in that experiment, or edit it back), wait-build, and move on. Append the outcome to the same line in `experiments.md`: the primary metric before and after, kept or reverted, and why.
6. After every kept change, re-measure the baseline against the current tree so later experiments compare against the kept state: `{{RUN}} bench {{NAME}} {{RUN_DIR}}/baseline-N --reps 5` and use that directory as the baseline from then on.
7. Stop after {{MAX_EXPERIMENTS}} experiments, after 3 consecutive reverts, or when a step's remaining cost is paint and compositing you cannot reach from the source. Before stopping, `{{RUN}} ab {{NAME}} {{RUN_DIR}}/final` measures baseline and candidate back to back (it stashes your changes, measures, restores them, measures) so drift cannot explain the gain.

Prefer changes with a mechanism you can name: fewer style writes on large subtrees, work moved off the hot path, listeners that coalesce per frame, cached rects, cheaper selectors, smaller DOM, removed layout thrash (a read after a write), avoided `:has()` invalidation, containment, `content-visibility`. Reject anything that changes rendering output, keyboard behaviour, the design system (no `box-shadow`, no decorative borders, tokens stay) or removes a feature. Garden rules: Preact and existing SCSS tokens; listeners inside `document.addEventListener('nav', ...)` with `window.addCleanup`; comments only for non-obvious constraints; kebab-case utility filenames; no new packages; no unit tests after code.

## Ownership

You may edit: {{OWNED}}
Read-only: everything else, including `{{HARNESS}}` (the harness lives in the main checkout, not your worktree), `quartz/styles/pages/bases.scss`, the emitters and the content. If a gain needs a shared file, write the exact change you would make into `{{RUN_DIR}}/proposals.md` with its expected effect and do not apply it.

## Allowed commands

`{{RUN}} ...`, `{{HARNESS}}/probe.py ...`, `rg`, `fd`, `sed -n`, `cat`, `head`, `tail`, `wc`, `ls`, `git status`, `git diff`, `git log`, `git show`, `git checkout -- <file>`, `pnpm exec tsgo --noEmit`, `pnpm exec oxlint <files>`, `pnpm exec oxfmt <files>`, `pnpm test <file>`, `curl` against `{{URL}}` only. No other network, no installs, no browsers of your own (the bench owns its tab), no sub-agents, no background processes.

## Final report

Your last message, in this order: outcome (`passed` when at least one change is kept with an ab-confirmed gain, `failed` when nothing could be kept, `incomplete` when stopped early); a table of kept changes with scenario, baseline and final medians and the ratio; what you tried and reverted; the files you changed (`git diff --stat`); baseline check failures you saw; and the next experiment you would run. Observed numbers only; the bench output is the evidence, not your estimate.
