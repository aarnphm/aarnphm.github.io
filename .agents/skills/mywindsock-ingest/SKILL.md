---
name: mywindsock-ingest
description: "Capture Aaron's personal myWindsock ride and run analyses through the signed-in browser and save validated per-Strava-ID JSON in Garden. Use for daily wind sync, manual per-activity ingestion, or backfill since Tour de Simcoe (2026-10-03)."
---

# myWindsock ingestion

Work in `/Users/aarnphm/workspace/garden`. Read its AGENTS.md and
`docs/mywindsock-ingestion.md`. The durable archive is
`content/triathlon/wind/<strava_id>.json`. This workflow is for Aaron's own
outdoor runs and bikes from **2026-10-03 inclusive**, the Tour de Simcoe day.

## Discover and seed

Inspect scoped Git status and preserve concurrent changes and staging.
For a daily sync, refresh the existing Strava cache with
`STRAVA_SYNC_REFRESH_DAYS=7 STRAVA_MAX_READ_REQUESTS=60 pnpm strava:sync`.
This uses the existing `.env` and token refresh contract. Do not print tokens.
If authentication or request limits prevent freshness, report that fact and
continue available cached activities without calling the cache current.
For a requested manual ID, use the existing cache first.

Run these local commands:

```sh
pnpm exec tsx quartz/scripts/ingest-mywindsock.ts seed
pnpm exec tsx quartz/scripts/ingest-mywindsock.ts queue
```

The queue uses Strava's local date, exact IDs, and the repo's sport
classification. It excludes virtual and trainer activities. Seed preserves
browser evidence and manual notes. Both default to the October 3 cutoff.
Use `--until YYYY-MM-DD` for a specified day; midnight runs cover the previous
local day. Process at most six `due: true` entries, newest first, per daily run.
An entry without a description report can still have a browser analysis.
Captured entries remain due until `ingestion.sections.allGraphs` is captured.
Once every graph is inventoried, unresolved mappings do not require a repeat
browser visit. A page menu may expose model outputs that remain semantically
unverified on a run.

## Capture one activity

Read [browser capture](references/browser-capture.md) before the first page.
Use the computer-use browser tool and its documented CDP capability. Reuse
the signed-in Chrome profile. The activity URL and View on Strava link must
identify the queued ID; `ride_time` must equal the cached recording start.
The importer enforces the timestamp and filename identity again.

Capture all available runtime arrays, aggregates, weather, every visible menu
graph, and every inline chart. Save a graph inventory with captured, empty,
unavailable, or failed state per entry. Preserve full native configurations,
series, axes, categories, polar labels, bands, and non-finite evidence.
Use the repository's versioned capture expression,
which removes account identifiers and records non-finite number ranges before
serialization. Keep chart pairs on their own axes. Save the exported download
and import it with:

```sh
pnpm exec tsx quartz/scripts/ingest-mywindsock.ts import <strava_id> <absolute-capture-path>
pnpm exec tsx quartz/scripts/ingest-mywindsock.ts graphs <strava_id> <absolute-graph-bundle-path>
```

The importer validates before an atomic save, computes a checksum, retains
context, and logs events. Repeating the same capture is a no-op. It refuses to
replace prior runtime evidence unless a human explicitly requested a reviewed
replacement using `--replace`. Do not use that flag in a daily run.
The `graphs` command adds graph evidence to an existing runtime capture and
preserves earlier successful graph data through incomplete retries.
Checkpoint each graph under `docs/verification/mywindsock-ingestion/artifacts/checkpoints/` so an
interrupted session can resume without repeating completed selections.
Never generate a performance change or AI experiment to make a conditional
view available. Record its actual requirement. Capture 3D view chart data;
its route geometry remains in the runtime capture.

## Partial results and retries

If a page lacks analysis, record the actual reason:

```sh
pnpm exec tsx quartz/scripts/ingest-mywindsock.ts attempt <strava_id> unavailable "Analysis is not available yet."
```

Use `failed` for browser access or adapter/schema failures. Attempts are kept
in an existing archive and an ignored local retry ledger at
`quartz/.quartz-cache/mywindsock-ingestion.json`. The queue retries on the next
Toronto day, including delayed reports. Do not create an evidence-free content
file, synthesize a successful capture, or change provider settings to get data.
If browser access or login prevents the whole batch, stop the browser portion
and state the required action. A stale lock requires checking the prior process
before removing that lock. Stop a command that exceeds five minutes and retain
its logs, following Garden's instructions.

Every imported browser record remains `captured`. Unknown units, selected
model settings, absent sections, and source disagreement remain explicit.
On a run, absent cycling CdA is unavailable. Internal run `cda` and `svm`
arrays are unverified; never present them as cycling CdA in square metres.
Do not derive wind-adjusted pace, true average temperature from a range,
variance definitions, or new graph data during ingestion.

## Verify and finish

Run `pnpm exec tsx quartz/scripts/validate-wind-archive.ts` after writes, then
run the queue again. Verify saved IDs, checksums, section states, and deferred
entries. Reference exact archive paths in the run result. Record any meaningful
additions or failures and the remaining queue count; an unchanged queue needs
no routine update. Keep changes local. Preserve source activities, descriptions,
account integrations, Git staging, and the existing dev process. Publishing and
graph development belong to separate explicitly requested work.
