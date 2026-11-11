import { isRecord, readNumber, readString } from '../../util/type-guards'
import {
  SWIM_STROKES,
  type AppleSwim,
  type AppleSwimInterval,
  type SwimLocation,
} from './apple-swims'
function swimLocation(value: unknown): SwimLocation | null {
  return value === 'pool' || value === 'openWater' ? value : null
}

export function parseAppleJsonSwims(raw: unknown): AppleSwim[] {
  const swims = isRecord(raw) ? raw.swims : undefined
  if (!Array.isArray(swims)) return []
  const out: AppleSwim[] = []
  for (const swim of swims) {
    if (!isRecord(swim)) continue
    const idValue = readString(swim, 'id')?.trim()
    const id = idValue ? idValue : null
    const date = readString(swim, 'date')?.slice(0, 10) ?? null
    const start = parseIsoSecond(swim.start)
    const end = parseIsoSecond(swim.end)
    const totalM = readNumber(swim, 'totalM')
    const laps = readNumber(swim, 'laps')
    const activeTimeS = num(swim.activeTimeS)
    const strokeCount = num(swim.strokeCount)
    const strokeTimeS = num(swim.strokeTimeS)
    const location = swimLocation(swim.location)
    const rawWaterTemperatureC = num(swim.waterTemperatureC)
    const waterTemperatureC =
      rawWaterTemperatureC != null && rawWaterTemperatureC >= -5 && rawWaterTemperatureC <= 60
        ? Math.round(rawWaterTemperatureC * 10) / 10
        : null
    const rawStrokes = isRecord(swim.strokes) ? swim.strokes : null
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date) || totalM == null || laps == null || !rawStrokes)
      continue
    const strokes: Record<string, number> = {}
    for (const [stroke, meters] of Object.entries(rawStrokes)) {
      const rounded = num(meters)
      if (rounded != null && rounded > 0) strokes[stroke] = Math.round(rounded)
    }
    const intervals: AppleSwimInterval[] = []
    const rawIntervals = Array.isArray(swim.intervals) ? swim.intervals : []
    for (const interval of rawIntervals) {
      if (!isRecord(interval)) continue
      const intervalStart = parseIsoSecond(interval.start)
      const intervalEnd = parseIsoSecond(interval.end)
      const distanceM = num(interval.distanceM)
      const startElapsedS = num(interval.startElapsedS)
      const endElapsedS = num(interval.endElapsedS)
      const durationS = num(interval.durationS)
      const intervalStrokeCount = num(interval.strokeCount)
      const intervalStrokeTimeS = num(interval.strokeTimeS)
      const strokeName = readString(interval, 'stroke')
      const stroke = SWIM_STROKES.find(candidate => candidate === strokeName) ?? null
      if (
        !intervalStart ||
        !intervalEnd ||
        Date.parse(intervalEnd) <= Date.parse(intervalStart) ||
        distanceM == null ||
        distanceM <= 0
      )
        continue
      intervals.push({
        start: intervalStart,
        end: intervalEnd,
        distanceM: Math.round(distanceM * 10) / 10,
        startElapsedS:
          startElapsedS != null && startElapsedS >= 0 ? Math.round(startElapsedS * 10) / 10 : null,
        endElapsedS:
          endElapsedS != null && endElapsedS > 0 ? Math.round(endElapsedS * 10) / 10 : null,
        durationS:
          durationS != null && durationS > 0
            ? Math.round(durationS * 10) / 10
            : Math.round(((Date.parse(intervalEnd) - Date.parse(intervalStart)) / 1000) * 10) / 10,
        strokeCount:
          intervalStrokeCount != null && intervalStrokeCount >= 0
            ? Math.round(intervalStrokeCount * 10) / 10
            : null,
        strokeTimeS:
          intervalStrokeTimeS != null && intervalStrokeTimeS > 0
            ? Math.round(intervalStrokeTimeS * 10) / 10
            : null,
        stroke,
      })
    }
    intervals.sort((a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end))
    out.push({
      id,
      date,
      start,
      end,
      totalM: Math.round(totalM),
      laps: Math.round(laps),
      activeTimeS: activeTimeS != null && activeTimeS >= 0 ? Math.round(activeTimeS) : null,
      strokeCount: strokeCount != null && strokeCount >= 0 ? Math.round(strokeCount) : null,
      strokeTimeS: strokeTimeS != null && strokeTimeS >= 0 ? Math.round(strokeTimeS) : null,
      strokes,
      intervals,
      location,
      waterTemperatureC,
    })
  }
  return out.sort(
    (a, b) =>
      (a.start ?? a.date).localeCompare(b.start ?? b.date) ||
      (a.id ?? '').localeCompare(b.id ?? ''),
  )
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function parseIsoSecond(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const time = Date.parse(value)
  if (!Number.isFinite(time)) return null
  return new Date(time).toISOString().replace('.000Z', 'Z')
}
