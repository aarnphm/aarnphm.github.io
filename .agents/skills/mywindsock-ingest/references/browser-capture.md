# Browser capture, classic adapter v1

All browser operations use `mcp__cua_repl.js`. On first use follow its exact
entry-point instructions. After a compacted browser session call
`await cua.rewriteDocumentation()`. Reuse the selected browser and read the
CDP capability documentation before calling `send`.

Open `https://mywindsock.com/activity/<id>/` in an ephemeral research tab.
Read its DOM snapshot. Wait for populated Moving Time, Distance and Totals,
using observations, without changing analysis start time, profile, or weather.
Check the visible View on Strava link and the cached ID/start. The runtime
`stravaID` has been `0` on real pages; courseID is a separate provider identity.

## Complete graph capture

Read `quartz/scripts/mywindsock-browser-session.mjs` before using it. Import
that local helper **inside cua_repl** and pass the documented tab/CDP handles.
It operates inspected chart controls, verifies selection and activity identity,
reads the native ZingChart `getdata` configuration, and checkpoints each graph.
Use one research tab sequentially so background animation and large maps do
not accumulate. Start only after the analysis is populated.

```js
const session =
  await import('file:///Users/aarnphm/workspace/garden/quartz/scripts/mywindsock-browser-session.mjs')
let bundle = await session.startGraphCapture(tab, cdp)
nodeRepl.write({ id: bundle.stravaId, menu: bundle.menu })
// Repeat bounded batches until every menu key is represented and none failed.
nodeRepl.write(await session.captureGraphBatch(tab, cdp, bundle, 6))
nodeRepl.write(await session.captureInlineGraphs(tab, cdp, bundle))
// Only when the canonical file lacks a runtime capture:
nodeRepl.write(await session.exportRuntimeCapture(cdp, bundle))
```

Each batch saves `docs/verification/mywindsock-ingestion/artifacts/checkpoints/<id>.json`.
Runtime export saves `<id>.runtime.json` beside it through CDP and local
filesystem writes, without printing the payload or requiring repeated native
downloads. Import runtime first, then use `ingest-mywindsock.ts graphs` for
the graph bundle. Existing runtime captures need only the `graphs` command.
After an interruption, load the checkpoint from disk, verify its ID/course/start
against the current page, and resume batches. Do not call `startGraphCapture`
over that partial checkpoint. Each batch stays below 45 seconds; never wait on
background `requestAnimationFrame`, which can remain suspended.

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

The long classic menu defeated locator scrolling during this capture. The
helper uses CDP to invoke each inspected DOM list item, then checks the selected
key and fresh UI snapshot. If selectors or rendering change, inspect the new
DOM and record a failed section instead of inventing an internal command.

## Manual fallback for the runtime envelope

Use Runtime.evaluate for developer inspection of the known report globals,
and Playwright locators for visible UI controls. The read-only chart expression
is:

```js
const chartReadExpression =
  "JSON.stringify(zingchart.exec('chartarea_target','getdata').graphset.map(g=>({type:g.type,...(g.title?.text?{title:g.title.text}:{}),series:g.series.map(s=>({label:s.text??null,values:s.values??[],scale:s.scales??null}))})))"
```

Initialize a REPL object `charts = { cda: [], gradient: [] }` for **each new
activity**. Never reuse the last activity's charts. Keep the getdata result's
`.result.value`, parsed as JSON. Do not print all chart values.

- If the visible **View the CdA Chart** link exists on a bike, click it and
  inspect the new chart state, then capture getdata into `charts.cda`.
- Click the visible **Feels Like Elevation Chart** link where available.
  Inspect the chart state and save its getdata into `charts.feelsLike`.
  This chart can contain `[elapsedMilliseconds, [lowerFeet, upperFeet]]`
  band pairs. Keep the bands intact.
- Open the visible selected chart item. On verified classic pages this is
  `#chart_menu_ul li.ulmenu_selected`. Read the expanded menu before choosing.
  There are two **Gradient** items: `virt_grade` is the wind-adjusted gradient;
  `grade` is actual gradient. After confirming those `data-value` attributes
  in the current DOM, choose `li[data-value="grade"]`, inspect the resulting
  **Gradient** chart, then save getdata into `charts.gradient`.
- Runs can omit the cycling CdA link. Leave `charts.cda` empty. Raw model
  arrays on a run do not establish cycling CdA availability.

If a selector fails, inspect current DOM/attributes instead of repeating it or
guessing an internal page command. Capture the available sections and mark
missing charts pending. Do not select a new model, click Save, Generate report,
Send Activity Report, or change dates/weather.

Read the local repository expression into the REPL (filesystem access here
is only for this task's source file):

```js
const captureScript = await (
  await import('node:fs/promises')
).readFile('/Users/aarnphm/workspace/garden/quartz/scripts/mywindsock-browser-capture.js', 'utf8')
const exported = await cdp.send('Runtime.evaluate', {
  expression: captureScript.replace('__MYWINDSOCK_CHARTS__', JSON.stringify(charts)),
  returnByValue: true,
})
nodeRepl.write(exported)
```

The expression returns small metadata and triggers a local JSON download with
a unique timestamp in its name. It does not persist page state. Never print
the full runtime payload or credentials. Read the exact returned filename in
`/Users/aarnphm/Downloads/` and pass that absolute path to the importer. A
Playwright download event has timed out despite a completed native download,
so verify the exact returned file on disk before declaring failure. Do not
pick an older similarly named file. Native download duplicate suffixes require
matching capturedAt and activity identity before import.

The adapter excludes `data.hex`, `data.windline_latlng`, `forecast.requested`,
and account/profile names and identifiers. It captures scoped dataarea,
weatherarea, and navigator labels, without serializing full HTML, cookies,
storage, account globals, request headers, or the surrounding page. The strict
importer also rejects nested authentication fields and unsupported shapes.

Known interpretation limits: analysis time is hours; accumulated distance is
kilometres; ground speed is km/h; `airdist_acc` is Air Penalty, distinct from
total air distance. Live CdA has fewer samples than the shared time array, so
use its chart's own coordinates. Wind bearings, full apparent-air semantics,
profile selection, forecast/historical provenance, and `cda_sd` are unresolved.
Preserve raw values and pending work when those limits affect an interpretation.
