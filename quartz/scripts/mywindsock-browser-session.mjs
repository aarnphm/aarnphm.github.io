import { mkdir, readFile, writeFile } from 'node:fs/promises'

// Run this module inside cua_repl, passing its documented tab and CDP handles.
const graphScript = await readFile(
  new URL('./mywindsock-browser-graph.js', import.meta.url),
  'utf8',
)
const captureScript = await readFile(
  new URL('./mywindsock-browser-capture.js', import.meta.url),
  'utf8',
)
const checkpointDirectory = '/Users/aarnphm/workspace/garden/quartz/.quartz-cache/mywindsock-graphs'

async function evaluate(cdp, expression) {
  const result = await cdp.send(
    'Runtime.evaluate',
    { expression, returnByValue: true },
    { timeoutMs: 10000 },
  )
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text)
  return result.result.value
}

export async function startGraphCapture(tab, cdp) {
  if (
    !(await tab.playwright.evaluate(() =>
      document.querySelector('#chart_menu_ul').classList.contains('ulmenu_active'),
    ))
  ) {
    await evaluate(cdp, 'document.querySelector("#chart_menu_ul li.ulmenu_selected").click(); true')
    await tab.playwright.domSnapshot()
  }
  const menu = await tab.playwright.evaluate(() =>
    Array.from(document.querySelectorAll('#chart_menu_ul li[data-value]'))
      .filter(e => getComputedStyle(e).display !== 'none')
      .map(e => ({ key: e.getAttribute('data-value'), label: e.textContent.trim() })),
  )
  const identity = JSON.parse(
    await evaluate(
      cdp,
      `JSON.stringify({
    adapterVersion:'mywindsock-graphs-browser-v1',capturedAt:new Date().toISOString(),
    pageUrl:location.href,stravaId:location.pathname.match(/^\\/activity\\/([1-9]\\d*)\\/$/)?.[1],
    viewOnStravaId:Array.from(document.querySelectorAll('a')).find(a=>a.textContent.includes('View on Strava'))?.href.match(/\\/activities\\/([1-9]\\d*)/)?.[1],
    providerCourseId:String(courseID),rideTimestamp:Number(ride_time)
  })`,
    ),
  )
  if (!identity.stravaId || identity.stravaId !== identity.viewOnStravaId || !menu.length)
    throw new Error('Activity identity or graph menu is missing')
  const bundle = { ...identity, menu, graphs: {} }
  await checkpoint(bundle)
  return bundle
}

async function checkpoint(bundle) {
  await mkdir(checkpointDirectory, { recursive: true })
  await writeFile(`${checkpointDirectory}/${bundle.stravaId}.json`, JSON.stringify(bundle))
}

async function assertCurrentActivity(cdp, bundle) {
  const identity = JSON.parse(
    await evaluate(
      cdp,
      'JSON.stringify({path:location.pathname,course:String(courseID),start:Number(ride_time)})',
    ),
  )
  if (
    identity.path !== `/activity/${bundle.stravaId}/` ||
    identity.course !== bundle.providerCourseId ||
    identity.start !== bundle.rideTimestamp
  )
    throw new Error('The browser activity changed during capture')
}

export async function captureGraphBatch(tab, cdp, bundle, limit = 4) {
  await assertCurrentActivity(cdp, bundle)
  const deferred = new Set([
    'interval_designer',
    'ai_power',
    'delta_compare',
    'delta_compare_avg',
    '3dcourse_virt',
  ])
  const pending = bundle.menu
    .filter(item => !bundle.graphs[item.key] || bundle.graphs[item.key].state === 'failed')
    .sort((a, b) => Number(deferred.has(a.key)) - Number(deferred.has(b.key)))
    .slice(0, limit)
  const results = []
  for (const item of pending) {
    try {
      await assertCurrentActivity(cdp, bundle)
      const previousConfiguration =
        item.key === 'ai_power'
          ? JSON.parse(
              await evaluate(
                cdp,
                graphScript.replace(
                  '__MYWINDSOCK_GRAPH__',
                  JSON.stringify({ label: item.label, source: 'menu', target: 'chartarea_target' }),
                ),
              ),
            ).configuration
          : null
      // The native locator's scrolling moves this menu during selection. Invoke the
      // inspected DOM control through CDP, then verify the resulting visible state.
      if (
        !(await tab.playwright.evaluate(() =>
          document.querySelector('#chart_menu_ul').classList.contains('ulmenu_active'),
        ))
      ) {
        await evaluate(
          cdp,
          'document.querySelector("#chart_menu_ul li.ulmenu_selected").click(); true',
        )
        await tab.playwright.domSnapshot()
      }
      await evaluate(
        cdp,
        `document.querySelector('#chart_menu_ul li[data-value='+${JSON.stringify(JSON.stringify(item.key))}+']').click(); true`,
      )
      const snapshot = await tab.playwright.domSnapshot()
      const selected = await tab.playwright.evaluate(() =>
        document.querySelector('#chart_menu_ul li.ulmenu_selected')?.getAttribute('data-value'),
      )
      if (selected !== item.key) throw new Error('The requested menu entry did not become selected')
      const graph = JSON.parse(
        await evaluate(
          cdp,
          graphScript.replace(
            '__MYWINDSOCK_GRAPH__',
            JSON.stringify({ label: item.label, source: 'menu', target: 'chartarea_target' }),
          ),
        ),
      )
      if (
        item.key === 'ai_power' &&
        (JSON.stringify(graph.configuration.graphset.map(g => g.series?.map(s => s.values))) ===
          JSON.stringify(previousConfiguration.graphset.map(g => g.series?.map(s => s.values))) ||
          !graph.configuration.graphset
            .flatMap(g => g.series ?? [])
            .some(s => s.values?.length && s.text && !String(s.text).includes('Elevation')))
      ) {
        graph.state = 'unavailable'
        graph.configuration = null
        graph.nonFiniteValues = []
        graph.note =
          'This menu entry does not render an AI pacing analysis in the activity view. The page offers a separate Try AI entrypoint; no analysis was generated.'
      }
      if (item.key === 'delta_compare' || item.key === 'delta_compare_avg') {
        const comparison = graph.configuration.graphset
          .flatMap(g => g.series ?? [])
          .filter(s => s.text === 'Diff' || s.text === 'Delta Variance')
        if (!comparison.some(s => s.values?.length)) {
          graph.state = 'unavailable'
          graph.note =
            'The page requires a Performance Change to show comparison data. The native empty comparison configuration is retained.'
        }
      }
      if (item.key === '3dcourse_virt')
        graph.note =
          'Native elevation plot for the 3D Perspective view. Full route geometry and provider virtual elevation are retained in browserCapture.runtime for the map.'
      bundle.graphs[item.key] = graph
      results.push({
        key: item.key,
        state: graph.state,
        series: graph.configuration?.graphset.map(g =>
          g.series?.map(s => ({ label: s.text ?? s.label ?? null, count: s.values?.length ?? 0 })),
        ),
        snapshotTail: snapshot.slice(-120),
      })
    } catch (error) {
      bundle.graphs[item.key] = {
        label: item.label,
        source: 'menu',
        state: 'failed',
        capturedAt: new Date().toISOString(),
        configuration: null,
        nonFiniteValues: [],
        note: String(error).slice(0, 500),
      }
      results.push({ key: item.key, state: 'failed', note: bundle.graphs[item.key].note })
    }
    await checkpoint(bundle)
  }
  return results
}

export async function exportRuntimeCapture(cdp, bundle) {
  await assertCurrentActivity(cdp, bundle)
  const cdaAvailable = await evaluate(
    cdp,
    'Array.from(document.querySelectorAll("#dataarea a")).some(a => a.textContent.includes("View the CdA Chart"))',
  )
  const legacyChart = key =>
    (bundle.graphs[key]?.configuration?.graphset ?? []).map(graph => ({
      type: graph.type,
      ...(typeof graph.title?.text === 'string' ? { title: graph.title.text } : {}),
      series: (graph.series ?? []).map(series => ({
        label: series.text ?? series.label ?? null,
        values: series.values ?? [],
        scale: typeof series.scales === 'string' ? series.scales : null,
      })),
    }))
  const charts = {
    cda: cdaAvailable ? legacyChart('cda') : [],
    gradient: legacyChart('grade'),
    feelsLike: legacyChart('virt_elev'),
  }
  const capture = JSON.parse(
    await evaluate(
      cdp,
      captureScript
        .replace('__MYWINDSOCK_CHARTS__', JSON.stringify(charts))
        .replaceAll('__MYWINDSOCK_OUTPUT__', JSON.stringify('return')),
    ),
  )
  const capturePath = `${checkpointDirectory}/${bundle.stravaId}.runtime.json`
  await mkdir(checkpointDirectory, { recursive: true })
  await writeFile(capturePath, JSON.stringify(capture))
  return {
    id: capture.stravaId,
    path: capturePath,
    analysisSamples: capture.runtime.series.time.length,
    series: Object.keys(capture.runtime.series).length,
  }
}

export async function captureInlineGraphs(tab, cdp, bundle) {
  await assertCurrentActivity(cdp, bundle)
  const targets = await tab.playwright.evaluate(() =>
    Array.from(document.querySelectorAll('svg[id]'))
      .filter(
        e =>
          e.id.endsWith('-svg') && !e.id.includes('-graph-id') && e.id !== 'chartarea_target-svg',
      )
      .map(e => ({
        id: e.id.slice(0, -4),
        label: e.parentElement.parentElement.parentElement.textContent.trim().slice(0, 120),
      })),
  )
  const results = []
  if (!targets.some(target => target.id === 'pointsgraph')) {
    const rankingText = await tab.playwright.evaluate(() =>
      document.querySelector('#summary-dashboard-premium')?.textContent.trim(),
    )
    if (rankingText?.includes('rankings yet')) {
      bundle.graphs['inline:pointsgraph'] = {
        label: 'Activity Weather Rankings',
        source: 'inline',
        state: 'unavailable',
        capturedAt: new Date().toISOString(),
        configuration: null,
        nonFiniteValues: [],
        note: rankingText,
      }
      results.push({ key: 'inline:pointsgraph', state: 'unavailable' })
      await checkpoint(bundle)
    }
  }
  for (const target of targets) {
    const key = `inline:${target.id}`
    try {
      bundle.graphs[key] = JSON.parse(
        await evaluate(
          cdp,
          graphScript.replace(
            '__MYWINDSOCK_GRAPH__',
            JSON.stringify({ label: target.label, source: 'inline', target: target.id }),
          ),
        ),
      )
    } catch (error) {
      bundle.graphs[key] = {
        label: target.label,
        source: 'inline',
        state: 'failed',
        capturedAt: new Date().toISOString(),
        configuration: null,
        nonFiniteValues: [],
        note: String(error).slice(0, 500),
      }
    }
    results.push({ key, state: bundle.graphs[key].state })
    await checkpoint(bundle)
  }
  return results
}
