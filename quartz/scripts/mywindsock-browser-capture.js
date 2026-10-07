// Evaluate only through the browser's CDP Runtime.evaluate on an inspected activity page.
// Replace the charts marker with the chart pairs collected through the page's read-only getdata command.
;(() => {
  const charts = __MYWINDSOCK_CHARTS__
  const id = location.pathname.match(/^\/activity\/([1-9]\d*)\/$/)?.[1]
  const link = Array.from(document.querySelectorAll('a')).find(a =>
    a.textContent.includes('View on Strava'),
  )
  const viewId = link?.href.match(
    /^https:\/\/(?:www\.)?strava\.com\/activities\/([1-9]\d*)\/?$/,
  )?.[1]
  if (!id || viewId !== id || type !== 'activity' || mode !== 'analyst')
    throw new Error('Activity page identity or mode changed')
  if (!Array.isArray(data.time) || !data.time.length || !Array.isArray(course) || !course.length)
    throw new Error('Analysis is not loaded')
  const nonFiniteValues = []
  const numeric = (value, path, index = null, component) => {
    if (value === null || Number.isFinite(value)) return value
    if (typeof value !== 'number') throw new Error('Unexpected numeric shape at ' + path)
    const kind = Number.isNaN(value) ? 'NaN' : value === Infinity ? 'Infinity' : '-Infinity'
    const previous = nonFiniteValues.at(-1)
    if (
      index !== null &&
      previous?.path === path &&
      previous.kind === kind &&
      previous.component === component &&
      previous.end === index - 1
    )
      previous.end = index
    else
      nonFiniteValues.push({
        path,
        kind,
        start: index,
        end: index,
        ...(component === undefined ? {} : { component }),
      })
    return null
  }
  const series = Object.fromEntries(
    Object.entries(data)
      .filter(([key]) => key !== 'hex' && key !== 'windline_latlng')
      .map(([key, values]) => {
        if (!Array.isArray(values)) throw new Error('Unexpected series shape: ' + key)
        return [
          key,
          values.map((value, index) =>
            Array.isArray(value)
              ? [value[0], numeric(value[1], '/runtime/series/' + key, index, 1)]
              : numeric(value, '/runtime/series/' + key, index),
          ),
        ]
      }),
  )
  const totals = Object.fromEntries(
    Object.entries(averages).map(([key, value]) => [
      key,
      numeric(value, '/runtime/averages/' + key),
    ]),
  )
  const pick = (object, keys) => Object.fromEntries(keys.map(key => [key, object[key]]))
  const capture = {
    adapterVersion: 'mywindsock-classic-browser-v1',
    transport: 'browser-runtime',
    valueKind: 'provider-analysis',
    conditionKind: 'unknown',
    capturedAt: new Date().toISOString(),
    pageUrl: 'https://mywindsock.com/activity/' + id + '/',
    stravaId: id,
    viewOnStravaId: viewId,
    providerCourseId: String(courseID),
    providerStravaId: Number(stravaID),
    rideTimestamp: Number(ride_time),
    pageType: type,
    pageMode: mode,
    runtime: { averages: totals, series, course, distancesM: distances, elevationM: elev },
    weather: forecast.map(f =>
      pick(f, [
        'latitude',
        'longitude',
        'timezone',
        'offset',
        'side_of_road',
        'hourly',
        'daily',
        'currently',
      ]),
    ),
    modelProfileCandidates: profiles.map(p =>
      pick(p, [
        'typeID',
        'weight',
        'rr',
        'dtl',
        'cda',
        'cda_up',
        'cda_down',
        'watts',
        'watts_up',
        'watts_down',
        'height',
        'ftp',
        'critical_power',
      ]),
    ),
    charts,
    observations: {
      summaryText: document.getElementById('dataarea')?.innerText ?? '',
      weatherText: document.getElementById('weatherarea')?.innerText ?? '',
      navigationText: document.getElementById('navigator_btn')?.textContent ?? '',
    },
    excludedFields: [
      'data.hex',
      'data.windline_latlng',
      'forecast.requested',
      'profile account identifiers and names',
    ],
    nonFiniteValues,
  }
  const body = JSON.stringify(capture, (_, value) => {
    if (typeof value === 'number' && !Number.isFinite(value))
      throw new Error('Unrecorded non-finite value outside runtime')
    return value
  })
  const output = typeof __MYWINDSOCK_OUTPUT__ === 'undefined' ? 'download' : __MYWINDSOCK_OUTPUT__
  if (output === 'return') return body
  const url = URL.createObjectURL(new Blob([body], { type: 'application/json' }))
  const anchor = document.createElement('a')
  anchor.href = url
  const filename = id + '.mywindsock-' + capture.capturedAt.replaceAll(':', '-') + '.json'
  anchor.download = filename
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  return JSON.stringify({
    id,
    filename,
    capturedAt: capture.capturedAt,
    analysisSamples: series.time.length,
    series: Object.keys(series).length,
    nonFiniteRanges: nonFiniteValues.length,
    chartGroups: Object.keys(charts),
  })
})()
