# Development commands for agents

Run commands from the owning package. Read its current `package.json` or task runner before relying on these descriptions.

| Command                           | Effect and scope                                                                                                                             |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `pnpm test <test-file>`           | Runs the selected Node tests through `tsx --test`. Use affected existing tests as well as new tests.                                         |
| `pnpm exec oxlint <files>`        | Lints selected files without applying fixes.                                                                                                 |
| `pnpm exec oxfmt --check <files>` | Checks formatting without rewriting files.                                                                                                   |
| `pnpm exec tsgo --noEmit`         | Checks the repository TypeScript project without emitting code. Distinguish unrelated baseline errors.                                       |
| `pnpm check`                      | Runs repository-wide formatting, lint, type, and test checks. Broader than a focused change usually needs.                                   |
| `pnpm dev`                        | Starts the development container workflow. Inspect the existing process before starting another instance.                                    |
| `pnpm swarm`                      | Starts `quartz/scripts/dev.ts`. Preserve an existing watcher.                                                                                |
| `pnpm prod`, `pnpm bundle`        | Builds the full Quartz site. Use the current watcher for ordinary verification.                                                              |
| `pnpm format`                     | Rewrites BibTeX and source formatting, applies lint fixes, and formats Python. Avoid it for scoped edits.                                    |
| `pnpm db:migrate:local`           | Mutates local development databases.                                                                                                         |
| `pnpm db:migrate`                 | Applies migrations to remote Cloudflare D1 databases. Requires authorization for that remote change.                                         |
| `pnpm cf:deploy`                  | Runs broad checks, remote database migrations, and a Worker deployment. Requires authorization covering those effects.                       |
| Provider sync or backfill scripts | May contact providers, replace caches, or update remote activities. Read the specific command and its flags before running it.               |
| `pnpm water-current:sync`         | Fetches NOAA LOOFS historical surface-current estimates for recent open-water swim routes and updates the local weather cache.               |
| `pnpm trainingpeaks:titles`       | Previews Strava sauna title/description changes for completed cardio workouts through the TrainingPeaks web API. Add `--write` to save them. |

`trainingpeaks:titles` uses the local Strava cache and explicit sauna links in `content/triathlon.md`. Refresh Strava with `pnpm strava:sync` when needed. It calls the authenticated API used by the TrainingPeaks website, which can change independently of this repository. TrainingPeaks limits its [published partner API](https://help.trainingpeaks.com/hc/en-us/articles/234441128-TrainingPeaks-API) to approved commercial developers.

`weather:sync` also runs `water-current:sync`. The current sync uses the existing Strava GPS stream and WeatherKit activity window to sample NOAA's uppermost Lake Ontario model layer. It keeps current speed and flow direction separate from atmospheric wind, preserves dry cells and missing hours as gaps, and records coverage and model provenance. It needs no credentials. Use `pnpm water-current:sync --id STRAVA_ID` for a bounded activity refresh, and `--force` to replace an existing matching estimate. Unsupported routes remain unavailable.

Set `TRAININGPEAKS_AUTH_COOKIE` in the ignored `.env` file to the value of your own signed-in account's `Production_tpAuth` cookie. In browser developer tools, find it under Application → Cookies → trainingpeaks.com. Copy only its value, without the cookie name or other cookies. The command exchanges it for a short-lived bearer token and renews that token once on HTTP 401. A session cookie eventually expires or can be revoked by signing out; refresh the saved value when authentication fails. Alternatively, set `TRAININGPEAKS_ACCESS_TOKEN` to an existing web API bearer token. The command runs through HTTP without an open browser or Apple Events permission. Never commit either credential.

The command fetches workouts for the selected Strava date range and verifies the signed-in athlete identity. Only completed, unplanned, stationary Other workouts with a unique date/duration match and a start time within two minutes are eligible. It fetches each workout again before saving, changes the title and description in a full-record PUT, and verifies the saved text and other workout fields with a fresh GET. Existing notes are retained, and the sync replaces its own description section on later runs. Ambiguous matches, locked workouts, conflicting source links, and records that stop matching before saving are skipped and produce a failing exit status. Missing matches are reported without creating workouts. An API or readback failure stops the run and preserves the results collected so far.

Use `--since YYYY-MM-DD`, `--until YYYY-MM-DD`, `--id STRAVA_ID`, or `--limit N` to narrow a run. `--sources` lists the selected Strava metadata without authentication or remote requests. Runs save a private report under `quartz/.quartz-cache/trainingpeaks-titles/`, including before/after text for proposed or attempted updates. `health:all` invokes this command with `--write`.

Do not delete generated files as a routine step after starting the watcher. Let the owning emitter handle its output. For browser checks, use the matching `build:ready` and successful HTTP response before inspecting source-rendered markup and interactions. DOM injection proves only a prototype unless the served implementation is separately verified.

On macOS, `pnpm swarm` uses polling for Wrangler's file watchers. Watching the large `public/` tree with native file watchers can exceed Darwin's `OPEN_MAX` descriptor range and cause esbuild to fail with `spawn EBADF`, even with a higher `ulimit`. Text files use a one-second polling interval; Chokidar retains its default binary-file interval. Explicit `CHOKIDAR_USEPOLLING` and `CHOKIDAR_INTERVAL` settings override these defaults. Quartz's watcher and Linux container launches retain their existing behavior.
