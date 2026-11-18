# Browser capture, classic adapter v1

All browser operations use the `cua` MCP server (`start` then `js` with the same
`session` ID). Read the returned `start` documentation first. Create an unfocused
task window with `browser.tabs.new({ url, newWindow: { left: 80, top: 80, width: 1280, height: 900 }, sessionName: 'mywindsock-ingest' })`, call `await tab.activate()`,
and close owned task tabs when done via `await cua.finish()`. Never modify or
close existing user tabs or groups.

Open `https://mywindsock.com/activity/<id>/` in the task tab and read its DOM
snapshot. Wait for populated Moving Time, Distance and Totals without changing
analysis start time, profile, or weather. Check the visible View on Strava link
and the cached exact string ID/start. On live pages, global `stravaID` is `0`
(serialized as `"0"` in new captures and retained as `0` in historical
checksum-covered captures), `courseID` is a separate positive provider identity,
and `forecast[*].side_of_road` may be `null` on runs.

## Complete graph and runtime capture

Read `quartz/scripts/mywindsock-browser-session.mjs` before using it. Import
that local helper inside `cua` `js` and pass the task `tab` handle
(`tab.playwright.evaluate` automatically slices payloads above 500 KB so large
captures stay below Helium CUA's 1 MiB broker frame limit). It operates
inspected chart controls, verifies selection and exact activity/course
identities, reads the native ZingChart `getdata` configuration, and checkpoints
each graph. Start only after the analysis is populated:

```js
var session =
  await import('file:///Users/aarnphm/workspace/garden/quartz/scripts/mywindsock-browser-session.mjs')
var bundle = await session.startGraphCapture(tab)
repl.write({ id: bundle.stravaId, menu: bundle.menu.length })
// Repeat bounded batches until every menu key is represented and none failed.
repl.write(await session.captureGraphBatch(tab, bundle, 6))
repl.write(await session.captureInlineGraphs(tab, bundle))
// Only when the canonical file lacks a runtime capture:
repl.write(await session.exportRuntimeCapture(tab, bundle))
```

Each batch saves `docs/verification/mywindsock-ingestion/artifacts/checkpoints/<id>.json`.
Runtime export saves `<id>.runtime.json` beside it via chunked page evaluation
and local filesystem writes, without printing the payload or relying on native
browser downloads (unsupported in Helium CUA 0.4). Import runtime first, then
use `ingest-mywindsock.ts graphs` for the graph bundle. Existing runtime
captures need only the `graphs` command. After an interruption, load the
checkpoint from disk, verify its ID/course/start against the current page, and
resume batches. Do not call `startGraphCapture` over that partial checkpoint.
Keep each `cua` `js` call well under its 60-second limit; never wait on
background `requestAnimationFrame`, which remains suspended on hidden tabs.

The observed menu has 33 entries. The two Gradient labels use distinct keys:
`virt_grade` is wind adjusted and `grade` is actual. Capture **every visible
entry**, even an empty distribution. Capture inline weather ranking and wind
rose charts. Preserve all graphsets, native axes, category labels, series and
non-finite JSON pointers. A chart can use elapsed time, distance, duration,
category bins, or polar bearings; leave those coordinates intact.

Delta/Delta Variance require a performance change when their comparison series
are empty. Mark that condition unavailable, retaining the empty native template.
The AI entry can retain the previous chart without generating a pacing result;
never label those stale series as AI data. 3D Perspective supplies an elevation
plot; its route geometry and virtual elevation are in the runtime capture.
Do not generate experiments. On a run, model CdA menu series remain unverified
and do not establish measured cycling CdA.

The long classic menu defeated locator scrolling during earlier captures. The
helper invokes each inspected DOM list item in-page, then checks the selected
key and fresh UI snapshot. If selectors or rendering change, inspect the new
DOM and record a failed section instead of inventing an internal command.

## Adapter exclusions and interpretation limits

The runtime adapter (`quartz/scripts/mywindsock-browser-capture.js`) excludes
`data.hex`, `data.windline_latlng`, `forecast.requested`, and account/profile
names and identifiers, and rejects unsafe numeric `stravaID` or `courseID`
globals before stringification. It captures scoped `dataarea`, `weatherarea`,
and navigator labels without serializing full HTML, cookies, storage, account
globals, request headers, or the surrounding page. The strict importer also
rejects nested authentication fields and unsupported shapes.

Known interpretation limits: analysis time is hours; accumulated distance is
kilometres; ground speed is km/h; `airdist_acc` is Air Penalty, distinct from
total air distance. Live CdA has fewer samples than the shared time array, so
use its chart's own coordinates. Wind bearings, full apparent-air semantics,
profile selection, forecast/historical provenance, and `cda_sd` are unresolved.
Preserve raw values and pending work when those limits affect an interpretation.
