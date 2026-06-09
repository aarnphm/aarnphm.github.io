export interface AppleDaily {
  date: string
  burnKcal: number | null
  activeKcal: number | null
  intakeKcal: number | null
  weightKg: number | null
}

export interface AppleCache {
  version?: number
  lastSync: number
  days: Record<string, AppleDaily>
}

export interface AppleRecord {
  date: string
  kind: 'active' | 'basal' | 'intake' | 'weight'
  value: number
  unit: string
  source: string
}

const LB_TO_KG = 0.45359237

const KIND_BY_TYPE: Record<string, AppleRecord['kind']> = {
  HKQuantityTypeIdentifierActiveEnergyBurned: 'active',
  HKQuantityTypeIdentifierBasalEnergyBurned: 'basal',
  HKQuantityTypeIdentifierDietaryEnergyConsumed: 'intake',
  HKQuantityTypeIdentifierBodyMass: 'weight',
}

export function matchAppleRecord(line: string): AppleRecord | null {
  if (!line.includes('<Record')) return null
  const type = /type="(HKQuantityTypeIdentifier\w+)"/.exec(line)?.[1]
  const kind = type ? KIND_BY_TYPE[type] : undefined
  if (!kind) return null
  const date = /startDate="(\d{4}-\d{2}-\d{2})/.exec(line)?.[1]
  const value = /value="([\d.]+)"/.exec(line)?.[1]
  if (!date || value === undefined) return null
  const unit = /unit="([^"]+)"/.exec(line)?.[1] ?? ''
  const source = /sourceName="([^"]*)"/.exec(line)?.[1] ?? ''
  return { date, kind, value: Number(value), unit, source }
}

const toKg = (value: number, unit: string): number =>
  unit.toLowerCase().startsWith('lb') ? value * LB_TO_KG : value

interface SourceAgg {
  sum: number
  count: number
  last: number
  unit: string
}

// Apple exports the same metric from several sources (Watch + iPhone + Oura + Strava),
// so summing every record triple-counts. Pick ONE source per day+metric — prefer the
// Apple Watch, else the source with the most samples — and use only its records.
function pickSource(sources: Map<string, SourceAgg>): SourceAgg | null {
  let watch: SourceAgg | null = null
  let best: SourceAgg | null = null
  for (const [name, agg] of sources) {
    if (/watch/i.test(name) && (!watch || agg.count > watch.count)) watch = agg
    if (!best || agg.count > best.count) best = agg
  }
  return watch ?? best
}

export function aggregateAppleRecords(records: AppleRecord[]): AppleDaily[] {
  const byDay = new Map<string, Map<AppleRecord['kind'], Map<string, SourceAgg>>>()
  for (const r of records) {
    let kinds = byDay.get(r.date)
    if (!kinds) {
      kinds = new Map()
      byDay.set(r.date, kinds)
    }
    let sources = kinds.get(r.kind)
    if (!sources) {
      sources = new Map()
      kinds.set(r.kind, sources)
    }
    const agg = sources.get(r.source) ?? { sum: 0, count: 0, last: 0, unit: r.unit }
    agg.sum += r.value
    agg.count += 1
    agg.last = r.value
    agg.unit = r.unit
    sources.set(r.source, agg)
  }
  const out: AppleDaily[] = []
  for (const [date, kinds] of byDay) {
    const at = (k: AppleRecord['kind']): SourceAgg | null => {
      const sources = kinds.get(k)
      return sources ? pickSource(sources) : null
    }
    const active = at('active')
    const basal = at('basal')
    const intake = at('intake')
    const weight = at('weight')
    const activeKcal = active ? Math.round(active.sum) : null
    const basalKcal = basal ? Math.round(basal.sum) : null
    const burnKcal =
      activeKcal != null || basalKcal != null ? (activeKcal ?? 0) + (basalKcal ?? 0) : null
    out.push({
      date,
      activeKcal,
      burnKcal,
      intakeKcal: intake ? Math.round(intake.sum) : null,
      weightKg: weight ? Math.round(toKg(weight.last, weight.unit) * 10) / 10 : null,
    })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export function parseAppleJson(raw: unknown): AppleDaily[] {
  const days = (raw as { days?: unknown })?.days
  if (!Array.isArray(days)) return []
  const out: AppleDaily[] = []
  for (const d of days) {
    if (typeof d !== 'object' || d === null) continue
    const r = d as Record<string, unknown>
    const date = typeof r.date === 'string' ? r.date.slice(0, 10) : null
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) continue
    const active = num(r.activeKcal)
    const basal = num(r.basalKcal)
    const burn =
      num(r.burnKcal) ?? (active != null || basal != null ? (active ?? 0) + (basal ?? 0) : null)
    const lbs = num(r.weightLbs)
    out.push({
      date,
      activeKcal: active != null ? Math.round(active) : null,
      burnKcal: burn != null ? Math.round(burn) : null,
      intakeKcal: num(r.intakeKcal) != null ? Math.round(num(r.intakeKcal)!) : null,
      weightKg:
        num(r.weightKg) != null
          ? Math.round(num(r.weightKg)! * 10) / 10
          : lbs != null
            ? Math.round(lbs * LB_TO_KG * 10) / 10
            : null,
    })
  }
  return out.sort((a, b) => a.date.localeCompare(b.date))
}

export function mergeAppleDay(prev: AppleDaily | undefined, next: AppleDaily): AppleDaily {
  if (!prev) return next
  return {
    date: next.date,
    burnKcal: next.burnKcal ?? prev.burnKcal,
    activeKcal: next.activeKcal ?? prev.activeKcal,
    intakeKcal: next.intakeKcal ?? prev.intakeKcal,
    weightKg: next.weightKg ?? prev.weightKg,
  }
}

export function latestAppleDate(days: Record<string, AppleDaily>): string | null {
  let latest: string | null = null
  for (const day of Object.values(days)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day.date)) continue
    if (!latest || day.date > latest) latest = day.date
  }
  return latest
}
