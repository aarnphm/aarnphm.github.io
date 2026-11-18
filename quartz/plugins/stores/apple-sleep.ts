import { isRecord, readString } from '../../util/type-guards'

export interface AppleSleepSample {
  id: string
  start: string
  end: string
  value: string
  sourceName: string
  sourceBundleIdentifier: string
}
export interface AppleSleepSession {
  id: string
  start: string
  end: string
  totalSleepS: number
  deepS: number | null
  lightS: number | null
  remS: number | null
  awakeS: number | null
  sourceName: string
  sourceBundleIdentifier: string
  stages?: { startTime: number; endTime: number; stage: 'light' | 'deep' | 'rem' | 'awake' }[]
}
export interface AppleSleepDay {
  date: string
  sleep: AppleSleepSession | null
  naps: AppleSleepSession[]
}
const values = new Set([
  'inBed',
  'awake',
  'asleepUnspecified',
  'asleepCore',
  'asleepDeep',
  'asleepREM',
])

export function parseAppleSleepSamples(raw: unknown): AppleSleepSample[] {
  if (!isRecord(raw) || !Array.isArray(raw.sleepSamples)) return []
  return raw.sleepSamples.flatMap(sample => {
    if (!isRecord(sample)) return []
    const start = readString(sample, 'start')
    const end = readString(sample, 'end')
    const value = readString(sample, 'value')
    if (!start || !end || !value || !values.has(value) || !(Date.parse(end) > Date.parse(start)))
      return []
    // A numeric offset is needed to assign the wake date independently of the importing host.
    if (!/(Z|[+-]\d\d:\d\d)$/.test(start) || !/(Z|[+-]\d\d:\d\d)$/.test(end)) return []
    return [
      {
        id: readString(sample, 'id') ?? `${start}|${end}|${value}`,
        start,
        end,
        value,
        sourceName: readString(sample, 'sourceName') ?? '',
        sourceBundleIdentifier: readString(sample, 'sourceBundleIdentifier') ?? '',
      },
    ]
  })
}

export function matchAppleSleepRecord(line: string): AppleSleepSample | null {
  if (!line.includes('<Record ') || !line.includes('HKCategoryTypeIdentifierSleepAnalysis'))
    return null
  const attr = (key: string): string => new RegExp(`\\b${key}="([^"]*)"`).exec(line)?.[1] ?? ''
  const timestamp = (value: string): string =>
    value.replace(' ', 'T').replace(/ ([+-]\d\d)(\d\d)$/, '$1:$2')
  const value = attr('value').replace('HKCategoryValueSleepAnalysis', '')
  const mapped =
    value === 'Asleep' ? 'asleepUnspecified' : value.slice(0, 1).toLowerCase() + value.slice(1)
  return (
    parseAppleSleepSamples({
      sleepSamples: [
        {
          start: timestamp(attr('startDate')),
          end: timestamp(attr('endDate')),
          value: mapped,
          sourceName: attr('sourceName'),
        },
      ],
    })[0] ?? null
  )
}

export function aggregateAppleSleep(samples: AppleSleepSample[]): Record<string, AppleSleepDay> {
  // In-bed envelopes may overlap several sessions and never count as measured sleep.
  const timed = samples
    .filter(s => s.value !== 'inBed')
    .map(s => ({ ...s, a: Date.parse(s.start), b: Date.parse(s.end) }))
    .sort((a, b) => a.a - b.a || a.b - b.b)
  const groups: (typeof timed)[] = []
  let end = -Infinity
  for (const sample of timed) {
    if (sample.a > end + 30 * 60_000) groups.push([])
    groups[groups.length - 1].push(sample)
    end = Math.max(end, sample.b)
  }
  const days: Record<string, AppleSleepDay> = {}
  for (const group of groups) {
    const sources = new Map<string, typeof timed>()
    for (const sample of group) {
      const key = `${sample.sourceBundleIdentifier}|${sample.sourceName}`
      sources.set(key, [...(sources.get(key) ?? []), sample])
    }
    const selected = [...sources.values()]
      .filter(s => s.some(r => r.value.startsWith('asleep')))
      .sort((a, b) => {
        const rank = (s: typeof timed): number =>
          /watch/i.test(s[0].sourceName) || s[0].sourceBundleIdentifier === 'com.apple.health'
            ? 1
            : 0
        return (
          rank(b) - rank(a) || b.length - a.length || a[0].sourceName.localeCompare(b[0].sourceName)
        )
      })[0]
    if (!selected) continue
    const boundaries = [...new Set(selected.flatMap(s => [s.a, s.b]))].sort((a, b) => a - b)
    const totals = new Map<string, number>()
    const stages: NonNullable<AppleSleepSession['stages']> = []
    for (let i = 1; i < boundaries.length; i++) {
      const a = boundaries[i - 1],
        b = boundaries[i]
      const covering = selected
        .filter(s => s.a <= a && s.b >= b)
        .sort(
          (x, y) =>
            Number(x.value === 'asleepUnspecified') - Number(y.value === 'asleepUnspecified') ||
            x.value.localeCompare(y.value),
        )
      const stage = covering[0]?.value
      if (stage) totals.set(stage, (totals.get(stage) ?? 0) + (b - a) / 1000)
      const name =
        stage === 'asleepCore'
          ? 'light'
          : stage === 'asleepDeep'
            ? 'deep'
            : stage === 'asleepREM'
              ? 'rem'
              : stage === 'awake'
                ? 'awake'
                : null
      if (name) stages.push({ startTime: a, endTime: b, stage: name })
    }
    const totalSleepS = [...totals].reduce(
      (sum, [stage, n]) => sum + (stage.startsWith('asleep') ? n : 0),
      0,
    )
    if (!totalSleepS) continue
    const first = selected[0],
      last = selected.reduce((a, b) => (a.b > b.b ? a : b))
    const date = last.end.slice(0, 10)
    const session: AppleSleepSession = {
      id: `apple:${first.sourceBundleIdentifier}:${first.start}`,
      start: first.start,
      end: last.end,
      totalSleepS,
      deepS: totals.get('asleepDeep') ?? null,
      lightS: totals.get('asleepCore') ?? null,
      remS: totals.get('asleepREM') ?? null,
      awakeS: totals.get('awake') ?? null,
      ...(stages.length ? { stages } : {}),
      sourceName: first.sourceName,
      sourceBundleIdentifier: first.sourceBundleIdentifier,
    }
    const day = (days[date] ??= { date, sleep: null, naps: [] })
    // HealthKit has no nap category. Treat short daytime sessions as inferred naps.
    const hour = Number(first.start.slice(11, 13))
    if (totalSleepS <= 3 * 3600 && hour >= 9 && hour < 20) day.naps.push(session)
    else if (!day.sleep || session.totalSleepS > day.sleep.totalSleepS) day.sleep = session
  }
  return days
}
