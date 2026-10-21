# base-perf: view performance hill climbs

A measured loop for the heavy browser views (Base views such as the cinematheque vault and the
antilibrary bookcase, the triathlon pages and the maps). A worker model edits a view in its own
worktree, the worktree's dev watcher rebuilds, the bench measures the real Helium profile, and the
worker keeps or reverts the change by the numbers. No node in the harness itself: the bench is a
uv script that drives Helium through the CUA broker (`~/.local/share/helium-cua`), which uses
TabFS for discovery and raw CDP for the page.

    harness/base-perf/scripts/run.sh pass            the four standard climbs in parallel
    harness/base-perf/scripts/run.sh bench vault out  measure one group against its worktree (or
                                                      the main watcher when there is no worktree)

## Pieces

| file             | role                                                                         |
| ---------------- | ---------------------------------------------------------------------------- |
| `scenarios.py`   | the fixed workloads: page, ready predicate, in-page script; groups per climb |
| `bench.py`       | loads each page in a task tab, runs the script, records metrics and a shot   |
| `stats.py`       | medians and ranges, ACCEPT/REJECT between two runs                           |
| `compare.py`     | `compare <baseline> <candidate>`                                             |
| `probe.py`       | evaluate an expression in a page the way the bench sees it                   |
| `lighthouse.py`  | Lighthouse 12 performance runs per page (pnpm dlx, Helium headless)          |
| `cua.py`         | the broker client (Unix socket, newline JSON, persistent REPL session)       |
| `worktree.sh`    | a worker's `w` worktree, linked node_modules, cloned cache, own dev watcher  |
| `climb.sh`       | one worker end to end: baseline, task, model run, patch                      |
| `task.md`        | the assignment template                                                      |
| `scripts/run.sh` | the entry point the workers call                                             |

## Metrics

Per scenario and rep: `ready_ms` (navigate to view ready), `interaction_ms` (sum of step wall
times; the primary metric), `sync_ms` (handler + style + layout cost of the dispatched events),
`task_ms`, `script_ms`, `layout_ms`, `style_ms` (Performance.getMetrics deltas over the script),
`longtasks` and `longtask_ms` (PerformanceObserver, buffered), `loaf` and `loaf_block_ms`
(long-animation-frame entries inside the steps and their blocking time), frames from a rAF loop
that runs during every step (`frames_dropped` against the display interval the prelude measures,
`fps_min` of the worst step, `worst_frame_ms`), DOM and memory after the script (`nodes`,
`layout_objects`, `listeners`, `documents`, `heap_mb` and `heap_total_mb` for V8, `heap_growth_mb`
over the script, `dom_heap_mb` for Blink's Oilpan heap through Runtime.getHeapUsage), and each
step's `frame_ms`, `fps`, `dropped` and `worst_ms`. The extension does not expose HeapProfiler or
Memory, so there is no forced GC: `nodes` can read double when the previous document is still
uncollected, which the medians absorb.

`bench --lighthouse N` adds N Lighthouse 12 runs per scenario after the task tab is closed
(`lighthouse.py`: performance category, desktop preset, simulated throttling, Helium headless in
its own instance): `lh_score`, `lh_fcp_ms`, `lh_lcp_ms`, `lh_si_ms`, `lh_tti_ms`, `lh_tbt_ms`,
`lh_fid_ms`, `lh_main_ms`, `lh_js_ms`, `lh_cls`, `lh_dom`, `lh_bytes_kb`, as medians over the N
runs. The climb's baseline and final `ab` use 2 runs; a single run's TBT swings about 20 percent.

A candidate is ACCEPT when its primary median beats the baseline's best rep, no guard metric
(`task_ms`, `longtask_ms`, `ready_ms`, `frames_dropped`, `nodes`, `listeners`, `heap_mb`) lands
above the baseline's worst rep, and no check broke. `dom_heap_mb` and `documents` are reported
only: detached documents from earlier reps stay in the Blink heap until collected (one run read 50
to 171 MB across reps), so a run-to-run median says little; a steady climb across the reps of one
run is the leak signal. `fps_min` and `lh_score`
read the other way (higher is better); the Lighthouse numbers are reported, not guards. Workers
re-baseline after every kept change and finish with `run.sh ab`, which measures candidate and
stashed baseline back to back.

## Workers

| climb       | group       | scenarios                                                   | worker          |
| ----------- | ----------- | ----------------------------------------------------------- | --------------- |
| `vault`     | `vault`     | vault, vault-directors, cards-films, kanban                 | sol (default)   |
| `views`     | `views`     | bookcase, cards-books, table-to-read, table-ideas           | astra (default) |
| `triathlon` | `triathlon` | tri-hub, tri-day, tri-analytics, tri-calendar, tri-training | sol (default)   |
| `maps`      | `maps`      | tri-maps, places-map                                        | astra (default) |

Workers: `sol` (codex exec, gpt-6.1-sol) and `astra` (codex exec, gpt-6-astra) at
`BP_CODEX_EFFORT` (medium) in a workspace-write sandbox; `fable` (`claude -p --bare`,
claude-fable-5-1 at `BP_FABLE_EFFORT`, on `CLAUDE_API_KEY` from `.env`: API credits, never the
subscription); `argon` (Gemini 4 Argon); `caw` (Cursor, `BP_CAW_MODEL`, grok-4.7-medium). Pick
per climb with `pass --workers vault,views,triathlon,maps` or `BP_WORKERS`, for example
`--workers fable,fable,argon,caw`. Every worker runs with no browser of its own; the bench owns
the only tab. Ports: 7400, 7500, 7600, 7700 (pages at port + 707). Other knobs:
`BP_BUDGET_USD`, `BP_MAX_EXPERIMENTS`, `BP_TIMEOUT`.

## After a pass

Each `<run-dir>/<name>/` holds `baseline/`, `cand-N/`, `experiments.md`, `events.ndjson`,
`last-message.txt` or `inspect.txt`, `changes.patch`, `status-{before,after}.txt`. Read the
report and `final/compare.md`, inspect the patch, reproduce the decisive bench against the
worktree if the gain matters, then `run.sh land <run-dir>/<name>` applies the patch to this
checkout's index (stage only; the commit stays with aarnphm). The worktree stays until removed
with `w --rm garden bp-<name>`.

## Caveats

The bench brings its tab to the front for every rep: Helium's active tab changes while a run is
on, and a background tab would get no animation frames. One bench runs at a time per machine
(`~/.local/state/helium-cua/base-perf.lock`): when a debugged tab closes, Helium drops the other
sessions' debugger leases, so two concurrent benches fail each other's reps with `STALE_HANDLE`.
The four climbs of a pass queue on that lock; their Lighthouse runs happen outside it. Mapbox scenarios depend on the GPU and
network; their checks pass on a drawn canvas, not on tiles. `base.test.ts` fails at baseline
(2026-10-21: `initializeVault` reads `document.body.dataset` in a DOM without a body); it is
recorded per climb in `baseline/checks.txt`.
