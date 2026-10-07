// Read a chart selected through the visible menu, or an observed inline chart ID.
;(() => {
  const selection = __MYWINDSOCK_GRAPH__
  const native = zingchart.exec(selection.target, 'getdata')
  if (!native?.graphset?.length) throw new Error('No graph configuration at ' + selection.target)
  const nonFiniteValues = []
  const clone = (value, path) => {
    if (typeof value === 'number' && !Number.isFinite(value)) {
      nonFiniteValues.push({
        path,
        kind: Number.isNaN(value) ? 'NaN' : value === Infinity ? 'Infinity' : '-Infinity',
      })
      return null
    }
    if (typeof value === 'function') return value.toString()
    if (Array.isArray(value)) return value.map((item, index) => clone(item, path + '/' + index))
    if (value && typeof value === 'object')
      return Object.fromEntries(
        Object.entries(value)
          .filter(([, item]) => item !== undefined)
          .map(([key, item]) => [
            key,
            clone(item, path + '/' + key.replaceAll('~', '~0').replaceAll('/', '~1')),
          ]),
      )
    return value
  }
  const configuration = clone({ graphset: native.graphset }, '')
  const hasValues = native.graphset.some(graph =>
    graph.series?.some(series => series.values?.length),
  )
  return JSON.stringify({
    label: selection.label,
    source: selection.source,
    state: hasValues ? 'captured' : 'empty',
    capturedAt: new Date().toISOString(),
    configuration,
    nonFiniteValues,
    note: null,
  })
})()
