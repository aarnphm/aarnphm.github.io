# Exact Strava IDs

Base: `7e5eb6cbebf756952869535c171c52f67aa1cff3`. Branch: `codex/strava-string-ids`.

Strava activity and athlete IDs use canonical positive decimal strings. API ingestion prefers `id_str`; only positive safe integers qualify for the legacy numeric fallback. A present malformed `id_str`, a missing identity, or an unsafe numeric identity fails ingestion. The migration does not attempt to reconstruct rounded numbers.

Safe legacy Strava caches migrate on read, including athlete identity and activity records. Schema version 8 stays unchanged so this conversion does not force a summary backfill. Stream, detail and geo keys stay exact, and activity keys must agree with their records. Unknown legacy athlete 0 becomes null until the authenticated profile supplies an identity. Empty syncs still fetch that profile.

Cache updates validate before replacing files. Full sync writes only after fetched identities have been validated. Detail refreshes acquire a cross-process lock, reread the latest cache, and replace it atomically; full sync retains newer detail refreshes. A crash can leave the lock file behind: subsequent updates fail rather than bypassing the lock. Unsafe legacy caches need an exact provider source before they can be migrated.

Tracking joins, manual activity references, power curves, analytics, previews, maps, anchors, planned workout links and myWindsock route projections retain exact IDs. Garmin, Wahoo and TrainingPeaks native identifier contracts stay unchanged. Manual sauna IDs retain their prior decimal value as strings. Historical myWindsock capture metadata accepts safe numeric IDs without rewriting checksum-covered capture bytes; new captures emit strings. Weather fingerprints retain the old decimal spelling in the hash input without numeric conversion, preserving existing weather data.

The default API host remains `www.strava.com/api/v3`. The [Strava changelog](https://developers.strava.com/docs/changelog/) announces string IDs and schedules the new host for January 4, 2027. [MetaAthlete](https://developers.strava.com/docs/reference/#api-models-MetaAthlete) also exposes `id_str`, with safe legacy fallback when absent.

## Replay

From the repository root with the existing dependencies:

```sh
STRAVA_ID_TEST_ARTIFACTS=docs/verification/strava-string-ids/artifacts ./node_modules/.bin/tsx --test quartz/util/strava-id.test.ts quartz/scripts/sync-strava-ids.e2e.test.ts quartz/util/strava-id-pipeline.e2e.test.ts
./node_modules/.bin/tsx --test --test-concurrency=4
./node_modules/.bin/oxlint
./node_modules/.bin/tsgo --noEmit --incremental false
pnpm_config_verify_deps_before_run=false pnpm run check
harness/token-budget/scripts/run.sh check
```

The pnpm setting prevents its automatic install step from replacing the worktree's shared dependency directory. It does not skip any `check` stage. The aggregate command stops at existing formatting failures, so lint, typecheck and tests were also run separately.

All API tests use a local HTTP fixture and private temporary caches. No live Strava mutation, site build, merge or deployment was performed.

The five new regression tests cover colliding numeric representations of `9007199254740992` and `9007199254740993`, string precedence, safe fallback, malformed and unsafe IDs, nested athletes, safe cache conversion and JSON roundtrips, reconciliation, exact detail/stream URLs, empty syncs, equipment sync, concurrent detail refreshes, lap IDs, tracking joins, moves, exclusions, manual sauna, preview/map serialization, power curves, analytics, planned links, myWindsock archives and route projection. Failure cases assert unchanged cache bytes.

## Results

| Check                   | Migration                                       | Same base commit                   |
| ----------------------- | ----------------------------------------------- | ---------------------------------- |
| New regression tests    | 5 passed                                        | Not present                        |
| Full suite              | 2,353 passed, 27 failed, 2 skipped              | 2,345 passed, 30 failed, 2 skipped |
| Full lint               | Passed                                          | Not separately rerun               |
| Typecheck               | 20 existing errors                              | Same 20 diagnostics                |
| Changed-file formatting | Passed                                          | Not applicable                     |
| Aggregate `pnpm check`  | Stops at 3 unrelated Markdown formatting errors | Same 3 files                       |
| Staged token budget     | Passed; no file over 5,000 tokens               | Not applicable                     |

Every failing test name in the migration also fails on baseline. The baseline additionally fails three OCR harness tests. `comparison.json` records the names. The formatting failures are `content/library/A Lesson Before Dying.md`, `content/library/Melancholy I-II.md`, and `content/library/The Gospel According to Jesus Christ.md`.

Ten oversized touched modules were split behind their existing entry points to satisfy the repository token budget. Whole-symbol comparison and an independent read-only review checked the extractions. The audit also identified and prompted fixes for preview validation, equipment-only sync, numeric ordering, Garmin ID guards, archive readers and lap identity handling.

Generated logs and response fixtures are in `artifacts/`: `full-tests.log`, `baseline-full-tests.log`, `comparison.json`, `typecheck.log`, `baseline-typecheck.log`, `pnpm-check.log`, `baseline-pnpm-check.log`, `cli-replay.json`, `pipeline-replay.json`, `module-proof.json`, and `token-budget.json`. The repository's artifact cleanup expires these outputs after 72 hours; the committed tests are the durable replay.
