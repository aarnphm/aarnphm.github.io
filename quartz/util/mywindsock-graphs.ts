import { isRecord } from './type-guards'

export interface MyWindsockArchiveReference {
  activityId: number
  capturedAt: string
  path: string
}

export interface MyWindsockGraphPoint {
  x: number | string
  y: number | null
  upper?: number | null
}

export interface MyWindsockGraphSeries {
  id: string
  label: string
  axis: string
  kind: 'line' | 'area' | 'bar' | 'range'
  points: MyWindsockGraphPoint[]
}

export interface MyWindsockGraphAxis {
  label: string
  minimum: number | null
  maximum: number | null
}

export interface MyWindsockGraphPlot {
  kind: 'cartesian' | 'radar' | 'pie' | 'gauge' | 'unsupported'
  xKind: 'elapsed' | 'duration' | 'category' | 'number'
  xLabel: string
  axes: Record<string, MyWindsockGraphAxis>
  series: MyWindsockGraphSeries[]
}

export interface MyWindsockGraph {
  key: string
  label: string
  state: 'captured' | 'empty' | 'unavailable' | 'failed'
  note: string | null
  plots: MyWindsockGraphPlot[]
  configuration: unknown
}

export interface MyWindsockGraphs {
  activityId: number
  sport: 'bike' | 'run'
  capturedAt: string
  cyclingCda: boolean
  graphs: MyWindsockGraph[]
}

const numeric = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null
const string = (value: unknown): string => (typeof value === 'string' ? value : '')
const record = (value: unknown): Record<string, unknown> => (isRecord(value) ? value : {})
const text = (value: unknown): string => string(record(value).text)

export const isMyWindsockArchiveReference = (
  value: unknown,
  activityId: number,
): value is MyWindsockArchiveReference =>
  isRecord(value) &&
  value.activityId === activityId &&
  typeof value.capturedAt === 'string' &&
  Number.isFinite(Date.parse(value.capturedAt)) &&
  value.path === `/triathlon/wind/${activityId}.json`

const graphPlot = (value: unknown, key: string, plotIndex: number): MyWindsockGraphPlot => {
  const plot = record(value)
  const nativeType = string(plot.type)
  const kind =
    nativeType === 'radar' || nativeType === 'pie' || nativeType === 'gauge'
      ? nativeType
      : ['mixed', 'line', 'area', 'bar', 'range', 'scatter'].includes(nativeType)
        ? 'cartesian'
        : 'unsupported'
  const rawSeries = Array.isArray(plot.series) ? plot.series : []
  const scaleX = record(plot['scale-x'])
  const transform = record(scaleX.transform)
  const dated = transform.type === 'date' && transform.all === '%H:%i:%s'
  const categoryLabels = record(plot['scale-k']).labels ?? scaleX.labels
  const labels = Array.isArray(categoryLabels) ? categoryLabels.map(string) : []
  const xKind =
    kind === 'radar' || kind === 'pie' || kind === 'gauge' || nativeType === 'bar'
      ? 'category'
      : dated
        ? key === 'pdc'
          ? 'duration'
          : 'elapsed'
        : 'number'
  const axes: Record<string, MyWindsockGraphAxis> = {}
  for (const [name, value] of Object.entries(plot)) {
    if (!/^scale-(y(?:-\d+)?|r)$/.test(name)) continue
    const axis = record(value)
    const limits = string(axis.values).split(':').map(Number)
    axes[name] = {
      label: text(axis.label),
      minimum: numeric(axis['min-value']) ?? (limits.length > 1 ? numeric(limits[0]) : null),
      maximum: numeric(axis['max-value']) ?? (limits.length > 1 ? numeric(limits[1]) : null),
    }
  }
  if (!axes['scale-y']) axes['scale-y'] = { label: '', minimum: null, maximum: null }
  const series = rawSeries.map((value, index): MyWindsockGraphSeries => {
    const source = record(value)
    const axis =
      string(source.scales)
        .split(',')
        .find(name => name.startsWith('scale-y')) ?? 'scale-y'
    const nativeKind = source.type ?? nativeType
    const seriesKind =
      nativeKind === 'area' || nativeKind === 'bar' || nativeKind === 'range' ? nativeKind : 'line'
    const label =
      string(source.text) ||
      axes[axis]?.label ||
      (axis === 'scale-y-2'
        ? 'Elevation'
        : key === 'virt_elev'
          ? 'Feels Like elevation'
          : `Series ${index + 1}`)
    const values = Array.isArray(source.values) ? source.values : []
    const points = values.map((value, pointIndex): MyWindsockGraphPoint => {
      if (Array.isArray(value)) {
        const x = typeof value[0] === 'string' ? value[0] : (numeric(value[0]) ?? pointIndex)
        if (Array.isArray(value[1]))
          return { x, y: numeric(value[1][0]), upper: numeric(value[1][1]) }
        return { x, y: numeric(value[1]) }
      }
      return { x: labels[pointIndex] || string(source.text) || pointIndex, y: numeric(value) }
    })
    return { id: `${key}:${plotIndex}:${index}`, label, axis, kind: seriesKind, points }
  })
  return {
    kind,
    xKind,
    xLabel:
      xKind === 'elapsed'
        ? 'Elapsed time'
        : xKind === 'duration'
          ? 'Duration'
          : text(scaleX.label) || (xKind === 'category' ? 'Provider bins' : 'Provider x'),
    axes,
    series,
  }
}

// Read only data fields from native configurations. Formatter/rule source remains inert JSON.
export function parseMyWindsockGraphs(value: unknown, activityId: number): MyWindsockGraphs | null {
  if (!isRecord(value) || value.schemaVersion !== 1 || value.provider !== 'mywindsock') return null
  const activity = record(value.activity)
  const capture = record(value.graphCapture)
  if (
    activity.stravaId !== String(activityId) ||
    (activity.sport !== 'bike' && activity.sport !== 'run') ||
    capture.stravaId !== String(activityId) ||
    capture.viewOnStravaId !== String(activityId) ||
    typeof capture.capturedAt !== 'string' ||
    !Number.isFinite(Date.parse(capture.capturedAt)) ||
    capture.pageUrl !== `https://mywindsock.com/activity/${activityId}/` ||
    !isRecord(capture.graphs)
  )
    return null
  const graphs: MyWindsockGraph[] = []
  for (const [key, value] of Object.entries(capture.graphs)) {
    if (!isRecord(value) || typeof value.label !== 'string') return null
    const state = value.state
    if (state !== 'captured' && state !== 'empty' && state !== 'unavailable' && state !== 'failed')
      return null
    const native = record(value.configuration).graphset
    const plots = Array.isArray(native) ? native : isRecord(native) ? [native] : []
    graphs.push({
      key,
      label:
        key === 'inline:summary_windrose_chart' ? 'Wind direction and headwind share' : value.label,
      state,
      note: typeof value.note === 'string' ? value.note : null,
      plots: plots.map((plot, index) => graphPlot(plot, key, index)),
      configuration: value.configuration,
    })
  }
  const legacyCda = record(record(value.browserCapture).charts).cda
  if (!graphs.length) return null
  const legacySeries = Array.isArray(legacyCda)
    ? legacyCda.flatMap(chart => {
        const series = record(chart).series
        return Array.isArray(series) ? series : []
      })
    : []
  const cyclingCda =
    activity.sport === 'bike' &&
    legacySeries.some(
      series => isRecord(series) && Array.isArray(series.values) && series.values.length > 0,
    )
  return { activityId, sport: activity.sport, capturedAt: capture.capturedAt, cyclingCda, graphs }
}

export interface MyWindsockTimelineSeries {
  id: string
  label: string
  unit: 'm' | 'km' | 'km/h' | '%' | 'deg' | 'm2'
  points: { elapsedS: number; value: number | null }[]
  domain?: [number, number]
}

// Native labelled chart pairs define their own elapsed coordinates and display units.
// No join by index, smoothing, resistance normalization, or inferred CdA uncertainty.
export function myWindsockTimelineSeries(archive: MyWindsockGraphs): MyWindsockTimelineSeries[] {
  const result: MyWindsockTimelineSeries[] = []
  const add = (
    key: string,
    select: (series: MyWindsockGraphSeries) => boolean,
    label: string | ((series: MyWindsockGraphSeries) => string),
    unit: MyWindsockTimelineSeries['unit'],
    multiplier = 1,
    domain?: [number, number],
  ): void => {
    const graph = archive.graphs.find(graph => graph.key === key)
    if (graph?.state !== 'captured') return
    for (const plot of graph.plots) {
      if (plot.kind !== 'cartesian' || plot.xKind !== 'elapsed') continue
      for (const series of plot.series.filter(select)) {
        const axisLabel = plot.axes[series.axis]?.label.toLowerCase() ?? ''
        let conversion = multiplier
        if (unit === 'm') {
          if (axisLabel === 'ft') conversion = 0.3048
          else if (axisLabel === 'm') conversion = 1
          else continue
        }
        if (unit === 'km/h') {
          if (axisLabel === 'mph') conversion = 1.609344
          else if (axisLabel === 'km/h' || axisLabel === 'kph') conversion = 1
          else continue
        }
        if (unit === 'km') {
          if (axisLabel === 'miles' || axisLabel === 'mi') conversion = 1.609344
          else if (axisLabel === 'km') conversion = 1
          else continue
        }
        if (
          key === 'watts' &&
          (plot.axes[series.axis]?.minimum !== 0 || plot.axes[series.axis]?.maximum !== 100)
        )
          continue
        let previous = -1
        const points: MyWindsockTimelineSeries['points'] = []
        for (const point of series.points) {
          if (typeof point.x !== 'number' || point.x < previous) return
          previous = point.x
          points.push({
            elapsedS: point.x / 1000,
            value: point.y == null ? null : point.y * conversion,
          })
        }
        if (points.filter(point => point.value != null).length < 2) continue
        const name = typeof label === 'function' ? label(series) : label
        result.push({
          id: `mywindsock-${key === 'watts' ? 'resistance-' : ''}${series.id}`,
          label: name,
          unit,
          points,
          ...(domain ? { domain } : {}),
        })
      }
    }
  }
  add('virt_elev', series => series.kind !== 'range', 'Feels Like elevation', 'm', 0.3048)
  add('virt_grade', series => series.label === 'Feels Like', 'Feels Like grade', '%')
  add('virt_grade', series => series.label === 'Actual', 'provider grade', '%')
  for (const [key, label] of [
    ['effective', 'air speed'],
    ['sidewind', 'crosswind magnitude'],
    ['windspeed', 'ambient wind'],
  ] satisfies [string, string][])
    add(
      key,
      series => series.axis === 'scale-y',
      series => (series.label === 'Wind Gusts' ? 'wind gusts' : label),
      'km/h',
      1.609344,
    )
  add('yaw', series => series.axis === 'scale-y', 'yaw', 'deg')
  add('wwatts', series => series.axis === 'scale-y', 'weather impact', '%')
  add('brake', series => series.axis === 'scale-y', 'braking', '%')
  add(
    'watts',
    () => true,
    series => series.label.replace('Acc Resistance', 'acceleration resistance'),
    '%',
    1,
    [0, 100],
  )
  if (archive.sport === 'bike')
    add('airdist_acc', series => series.axis === 'scale-y', 'total air penalty', 'km', 1.609344)
  if (archive.cyclingCda)
    add(
      'cda',
      series => series.label === 'CdA' || series.label === 'Live CdA',
      series => series.label,
      'm2',
    )
  return result
}
