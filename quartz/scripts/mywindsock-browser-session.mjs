import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

// Run this module inside Helium CUA (`cua` MCP `js`), passing its claimed/task `tab` handle.
const graphScript = await readFile(
  new URL('./mywindsock-browser-graph.js', import.meta.url),
  'utf8',
)
const captureScript = await readFile(
  new URL('./mywindsock-browser-capture.js', import.meta.url),
  'utf8',
)
const defaultCheckpointDirectory = fileURLToPath(
  new URL('../../docs/verification/mywindsock-ingestion/artifacts/checkpoints', import.meta.url),
)

const INPUT_CHUNK_CHARS = 400_000
const OUTPUT_CHUNK_CHARS = 500_000

const isCdp = candidate => Boolean(candidate && typeof candidate.send === 'function')
const resolveChannel = (tab, maybeCdp) => (isCdp(maybeCdp) ? maybeCdp : isCdp(tab) ? tab : tab)
const resolveDir = options => options?.checkpointDirectory ?? defaultCheckpointDirectory

async function evaluate(channel, expression) {
  if (isCdp(channel)) {
    const result = await channel.send(
      'Runtime.evaluate',
      { expression, returnByValue: true },
      { timeoutMs: 10000 },
    )
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text)
    return result.result.value
  }
  const pw = channel?.playwright
  if (!pw?.evaluate) throw new Error('Expected a Helium CUA tab with tab.playwright.evaluate')

  let inputKey = null
  try {
    let runExpr = expression
    if (expression.length > INPUT_CHUNK_CHARS) {
      inputKey = '__mywindsock_in_' + Math.random().toString(36).slice(2)
      await pw.evaluate(key => {
        window[key] = ''
      }, inputKey)
      for (let offset = 0; offset < expression.length; offset += INPUT_CHUNK_CHARS) {
        await pw.evaluate(
          ({ key, slice }) => {
            window[key] += slice
          },
          { key: inputKey, slice: expression.slice(offset, offset + INPUT_CHUNK_CHARS) },
        )
      }
      runExpr = `window[${JSON.stringify(inputKey)}]`
    }

    const initial = await pw.evaluate(
      ({ expr, fromKey, maxChars }) => {
        const source = (fromKey ? window[fromKey] : expr).replace(/^(?:\s|\/\/[^\n]*\n|;)+/, '')
        const value = Function(`"use strict"; return (\n${source}\n)`)()
        if (typeof value !== 'string' || value.length <= maxChars) return { done: true, value }
        const outKey = '__mywindsock_out_' + Math.random().toString(36).slice(2)
        window[outKey] = value
        return { done: false, key: outKey, length: value.length, head: value.slice(0, maxChars) }
      },
      { expr: inputKey ? '' : runExpr, fromKey: inputKey, maxChars: OUTPUT_CHUNK_CHARS },
    )
    if (initial.done) return initial.value
    try {
      let assembled = initial.head
      for (let offset = OUTPUT_CHUNK_CHARS; offset < initial.length; offset += OUTPUT_CHUNK_CHARS) {
        assembled += await pw.evaluate(
          ({ key, start, size }) => window[key].slice(start, start + size),
          { key: initial.key, start: offset, size: OUTPUT_CHUNK_CHARS },
        )
      }
      return assembled
    } finally {
      await pw.evaluate(key => {
        delete window[key]
      }, initial.key)
    }
  } finally {
    if (inputKey) {
      await pw.evaluate(key => {
        delete window[key]
      }, inputKey)
    }
  }
}

const courseIdGuardExpression = `(typeof courseID==='string'?/^[1-9]\\d*$/.test(courseID):Number.isSafeInteger(courseID)&&courseID>0)?String(courseID):(()=>{throw new Error('Unexpected provider courseID')})()`

export async function startGraphCapture(tab, cdpOrOptions, maybeOptions) {
  const channel = resolveChannel(tab, cdpOrOptions)
  const options = isCdp(cdpOrOptions) ? maybeOptions : cdpOrOptions
  if (
    !(await tab.playwright.evaluate(() =>
      document.querySelector('#chart_menu_ul').classList.contains('ulmenu_active'),
    ))
  ) {
    await evaluate(
      channel,
      '(document.querySelector("#chart_menu_ul li.ulmenu_selected").click(), true)',
    )
    await tab.playwright.domSnapshot()
  }
  const menu = await tab.playwright.evaluate(() =>
    Array.from(document.querySelectorAll('#chart_menu_ul li[data-value]'))
      .filter(e => getComputedStyle(e).display !== 'none')
      .map(e => ({ key: e.getAttribute('data-value'), label: e.textContent.trim() })),
  )
  const identity = JSON.parse(
    await evaluate(
      channel,
      `JSON.stringify({
    adapterVersion:'mywindsock-graphs-browser-v1',capturedAt:new Date().toISOString(),
    pageUrl:location.href,stravaId:location.pathname.match(/^\\/activity\\/([1-9]\\d*)\\/$/)?.[1],
    viewOnStravaId:Array.from(document.querySelectorAll('a')).find(a=>a.textContent.includes('View on Strava'))?.href.match(/\\/activities\\/([1-9]\\d*)/)?.[1],
    providerCourseId:${courseIdGuardExpression},rideTimestamp:Number(ride_time)
  })`,
    ),
  )
  if (!identity.stravaId || identity.stravaId !== identity.viewOnStravaId || !menu.length)
    throw new Error('Activity identity or graph menu is missing')
  const bundle = { ...identity, menu, graphs: {} }
  await checkpoint(bundle, resolveDir(options))
  return bundle
}

async function checkpoint(bundle, directory = defaultCheckpointDirectory) {
  await mkdir(directory, { recursive: true })
  await writeFile(`${directory}/${bundle.stravaId}.json`, JSON.stringify(bundle))
}

async function assertCurrentActivity(channel, bundle) {
  const identity = JSON.parse(
    await evaluate(
      channel,
      `JSON.stringify({path:location.pathname,course:${courseIdGuardExpression},start:Number(ride_time)})`,
    ),
  )
  if (
    identity.path !== `/activity/${bundle.stravaId}/` ||
    identity.course !== bundle.providerCourseId ||
    identity.start !== bundle.rideTimestamp
  )
    throw new Error('The browser activity changed during capture')
}

export async function captureGraphBatch(
  tab,
  bundleOrCdp,
  limitOrBundle = 4,
  limitOrOptions,
  maybeOptions,
) {
  const hasCdp = isCdp(bundleOrCdp)
  const channel = resolveChannel(tab, hasCdp ? bundleOrCdp : undefined)
  const bundle = hasCdp ? limitOrBundle : bundleOrCdp
  const limit =
    typeof (hasCdp ? limitOrOptions : limitOrBundle) === 'number'
      ? hasCdp
        ? limitOrOptions
        : limitOrBundle
      : 4
  const options = hasCdp
    ? typeof limitOrOptions === 'object'
      ? limitOrOptions
      : maybeOptions
    : typeof limitOrBundle === 'object'
      ? limitOrBundle
      : limitOrOptions
  const dir = resolveDir(options)

  await assertCurrentActivity(channel, bundle)
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
      await assertCurrentActivity(channel, bundle)
      const previousConfiguration =
        item.key === 'ai_power'
          ? JSON.parse(
              await evaluate(
                channel,
                graphScript.replace(
                  '__MYWINDSOCK_GRAPH__',
                  JSON.stringify({ label: item.label, source: 'menu', target: 'chartarea_target' }),
                ),
              ),
            ).configuration
          : null
      // The native locator's scrolling moves this menu during selection. Invoke the
      // inspected DOM control in-page, then verify the resulting visible state.
      if (
        !(await tab.playwright.evaluate(() =>
          document.querySelector('#chart_menu_ul').classList.contains('ulmenu_active'),
        ))
      ) {
        await evaluate(
          channel,
          '(document.querySelector("#chart_menu_ul li.ulmenu_selected").click(), true)',
        )
        await tab.playwright.domSnapshot()
      }
      await evaluate(
        channel,
        `(document.querySelector('#chart_menu_ul li[data-value='+${JSON.stringify(JSON.stringify(item.key))}+']').click(), true)`,
      )
      const snapshot = await tab.playwright.domSnapshot()
      const selected = await tab.playwright.evaluate(() =>
        document.querySelector('#chart_menu_ul li.ulmenu_selected')?.getAttribute('data-value'),
      )
      if (selected !== item.key) throw new Error('The requested menu entry did not become selected')
      const graph = JSON.parse(
        await evaluate(
          channel,
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
    await checkpoint(bundle, dir)
  }
  return results
}

export async function exportRuntimeCapture(
  tabOrCdp,
  bundleOrCdp,
  maybeBundleOrOptions,
  maybeOptions,
) {
  const secondIsCdp = isCdp(bundleOrCdp)
  const channel = secondIsCdp ? bundleOrCdp : tabOrCdp
  const bundle = secondIsCdp ? maybeBundleOrOptions : bundleOrCdp
  const options = secondIsCdp ? maybeOptions : maybeBundleOrOptions
  const dir = resolveDir(options)

  await assertCurrentActivity(channel, bundle)
  const cdaAvailable = await evaluate(
    channel,
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
      channel,
      captureScript
        .replace('__MYWINDSOCK_CHARTS__', JSON.stringify(charts))
        .replaceAll('__MYWINDSOCK_OUTPUT__', JSON.stringify('return')),
    ),
  )
  const capturePath = `${dir}/${bundle.stravaId}.runtime.json`
  await mkdir(dir, { recursive: true })
  await writeFile(capturePath, JSON.stringify(capture))
  return {
    id: capture.stravaId,
    path: capturePath,
    analysisSamples: capture.runtime.series.time.length,
    series: Object.keys(capture.runtime.series).length,
  }
}

export async function captureInlineGraphs(tab, bundleOrCdp, maybeBundleOrOptions, maybeOptions) {
  const hasCdp = isCdp(bundleOrCdp)
  const channel = resolveChannel(tab, hasCdp ? bundleOrCdp : undefined)
  const bundle = hasCdp ? maybeBundleOrOptions : bundleOrCdp
  const options = hasCdp ? maybeOptions : maybeBundleOrOptions
  const dir = resolveDir(options)

  await assertCurrentActivity(channel, bundle)
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
      await checkpoint(bundle, dir)
    }
  }
  for (const target of targets) {
    const key = `inline:${target.id}`
    try {
      bundle.graphs[key] = JSON.parse(
        await evaluate(
          channel,
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
    await checkpoint(bundle, dir)
  }
  return results
}
