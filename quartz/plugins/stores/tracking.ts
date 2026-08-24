export interface TrackEntry {
  date: string
  weightLbs: number | null
  weightKg: number | null
  windKph: number | null
  windDir: string | null
  race: boolean
  event: string | null
}

export interface RaceEvent {
  date: string
  event: string | null
}

export interface ManualFuelingEntry {
  date: string
  activityId: number
  caloriesConsumed: number
}

export interface StrengthSet {
  repetitions: number | null
  durationS: number | null
  weightKg: number | null
}

export interface StrengthExercise {
  name: string
  setCount: number
  repetitions: number | null
  durationS: number | null
  sets: StrengthSet[]
}

export interface ManualStrengthEntry {
  date: string
  activityId: number
  volumeKg: number | null
  totalSets: number | null
  totalReps: number | null
  exercises: StrengthExercise[]
}

export interface TrainingExclusion {
  date: string
  activityId: number
}

export type SaunaCooldown = 'natural' | 'cold plunge'

export interface ManualSaunaEntry {
  id: number
  title: string | null
  date: string
  time: string
  durationS: number
  temperatureC: number
  humidityPct: number
  cooldown: SaunaCooldown
  heatTrainingLoad: number | null
}

export interface TrackingData {
  days: TrackEntry[]
  races: RaceEvent[]
  fueling: ManualFuelingEntry[]
  strength: ManualStrengthEntry[]
  sauna: ManualSaunaEntry[]
  trainingExclusions: TrainingExclusion[]
}

export interface ParsedTrackingBlock {
  day: TrackEntry
  fueling: ManualFuelingEntry | null
  strength: ManualStrengthEntry | null
  sauna: ManualSaunaEntry | null
  trainingExclusion: TrainingExclusion | null
}

const LB_TO_KG = 0.45359237
const MANUAL_ACTIVITY_ID_OFFSET = 8_000_000_000_000

const massKg = (value: number, unit: string): number =>
  Math.round(value * (unit.toLowerCase().startsWith('lb') ? LB_TO_KG : 1) * 1_000) / 1_000

const parseStrengthDuration = (value: string): number | null => {
  const match = /^(?:(\d+)m)?\s*(?:(\d+)s)?$/i.exec(value.trim())
  if (!match || (match[1] == null && match[2] == null)) return null
  return Number(match[1] ?? 0) * 60 + Number(match[2] ?? 0)
}

const parseStrengthEffort = (value: string): StrengthSet | null => {
  const parts = value.split(/\s*@\s*/)
  if (parts.length > 2) return null
  const repetitions = /^(\d+)\s+reps?$/i.exec(parts[0])
  const durationS = repetitions ? null : parseStrengthDuration(parts[0])
  if (!repetitions && durationS == null) return null
  let weightKg: number | null = null
  if (parts[1] != null) {
    const weight = /^(\d+(?:\.\d+)?)\s*(kg|lbs?)$/i.exec(parts[1])
    if (!weight) return null
    weightKg = massKg(Number(weight[1]), weight[2])
  }
  return { repetitions: repetitions ? Number(repetitions[1]) : null, durationS, weightKg }
}

const parseStrengthExercise = (value: string): StrengthExercise | null => {
  const [rawName, ...rawEfforts] = value.split('|').map(part => part.trim())
  if (!rawName || rawEfforts.length === 0 || rawEfforts.some(effort => !effort)) return null
  if (rawEfforts.length === 1) {
    const aggregate = /^(\d+)\s+sets?(?:\s*\/\s*(.+))?$/i.exec(rawEfforts[0])
    if (aggregate) {
      const effort = aggregate[2] ? parseStrengthEffort(aggregate[2]) : null
      if (aggregate[2] && !effort) return null
      return {
        name: rawName,
        setCount: Number(aggregate[1]),
        repetitions: effort?.repetitions ?? null,
        durationS: effort?.durationS ?? null,
        sets: [],
      }
    }
  }
  const sets = rawEfforts.map(parseStrengthEffort)
  if (sets.some(set => set == null)) return null
  const parsedSets = sets.filter((set): set is StrengthSet => set != null)
  const repetitions = parsedSets.reduce((total, set) => total + (set.repetitions ?? 0), 0)
  const durationS = parsedSets.reduce((total, set) => total + (set.durationS ?? 0), 0)
  return {
    name: rawName,
    setCount: parsedSets.length,
    repetitions: parsedSets.some(set => set.repetitions != null) ? repetitions : null,
    durationS: parsedSets.some(set => set.durationS != null) ? durationS : null,
    sets: parsedSets,
  }
}

const parseSaunaDuration = (value: string): number | null => {
  const match = /^(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)$/i.exec(value.trim())
  if (!match) return null
  const seconds = Number(match[1]) * 60
  return Number.isSafeInteger(seconds) && seconds > 0 ? seconds : null
}

const parseSaunaTemperatureC = (value: string): number | null => {
  const match = /^(-?\d+(?:\.\d+)?)\s*°?\s*([fc])$/i.exec(value.trim())
  if (!match) return null
  const temperature = Number(match[1])
  if (!Number.isFinite(temperature)) return null
  const celsius = match[2].toLowerCase() === 'f' ? ((temperature - 32) * 5) / 9 : temperature
  return Math.round(celsius * 1_000) / 1_000
}

export const manualSaunaActivityId = (date: string, time: string): number =>
  MANUAL_ACTIVITY_ID_OFFSET + Number(`${date.replaceAll('-', '')}${time.replace(':', '')}`)

const parseManualSauna = (body: Readonly<Record<string, string>>): ManualSaunaEntry | null => {
  if (body.activity?.toLowerCase() !== 'sauna') return null
  const title = body.title?.trim() || null
  const date = body.date?.slice(0, 10)
  const time = body.time
  const durationS = parseSaunaDuration(body.duration ?? '')
  const temperatureC = parseSaunaTemperatureC(body.temperature ?? '')
  const humidity = /^(\d+(?:\.\d+)?)\s*%$/.exec(body.humidity ?? '')
  const humidityPct = humidity ? Number(humidity[1]) : NaN
  const cooldown = body.cooldown?.toLowerCase()
  const heatTrainingLoad = body.htl == null || body.htl === '' ? null : Number(body.htl)
  if (
    !date ||
    !/^\d{2}:\d{2}$/.test(time ?? '') ||
    Number(time.slice(0, 2)) > 23 ||
    Number(time.slice(3, 5)) > 59 ||
    durationS == null ||
    temperatureC == null ||
    !Number.isFinite(humidityPct) ||
    humidityPct < 0 ||
    humidityPct > 100 ||
    (cooldown !== 'natural' && cooldown !== 'cold plunge') ||
    (heatTrainingLoad != null && (!Number.isFinite(heatTrainingLoad) || heatTrainingLoad < 0))
  )
    return null
  return {
    id: manualSaunaActivityId(date, time),
    title,
    date,
    time,
    durationS,
    temperatureC,
    humidityPct,
    cooldown,
    heatTrainingLoad,
  }
}

export function parseTrackingMeta(meta: string | null | undefined): {
  race: boolean
  event: string | null
} {
  let race = false
  let event: string | null = null
  if (!meta) return { race, event }
  const re = /(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/g
  let m: RegExpExecArray | null
  while ((m = re.exec(meta)) !== null) {
    const key = m[1].toLowerCase()
    const val = m[2] ?? m[3] ?? m[4] ?? ''
    if (key === 'race') race = val === 'true' || val === '1' || val === 'yes'
    else if (key === 'event') event = val
  }
  return { race, event }
}

export function parseTrackingBlock(
  meta: string | null | undefined,
  value: string,
): ParsedTrackingBlock | null {
  const body: Record<string, string> = {}
  const exerciseValues: string[] = []
  for (const line of value.split('\n')) {
    const idx = line.indexOf(':')
    if (idx < 0) continue
    const k = line.slice(0, idx).trim().toLowerCase()
    const v = line.slice(idx + 1).trim()
    if (k === 'exercise') exerciseValues.push(v)
    else if (k) body[k] = v
  }
  const date = body.date
  if (!date || !/^\d{4}-\d{2}-\d{2}/.test(date)) return null
  const wl = body.weight != null ? Number(body.weight) : NaN
  const weightLbs = Number.isFinite(wl) ? wl : null
  const weightKg = weightLbs != null ? Math.round(weightLbs * LB_TO_KG * 10) / 10 : null
  const wind =
    /^(\d+(?:\.\d+)?)\s*(kph|km\/h|mph)?(?:\s+(?:\d+(?:\.\d+)?)?\s*([nsew]{1,3})?)?$/i.exec(
      body.wind ?? '',
    )
  const windKph = wind
    ? Math.round(Number(wind[1]) * (wind[2]?.toLowerCase() === 'mph' ? 1.609 : 1))
    : null
  const windDir = wind?.[3] ? wind[3].toUpperCase() : null
  const { race, event } = parseTrackingMeta(meta)
  const day = { date: date.slice(0, 10), weightLbs, weightKg, windKph, windDir, race, event }
  const activityId = Number(body.activity)
  const caloriesConsumed = Number(body.fueling)
  const fueling =
    Number.isSafeInteger(activityId) &&
    activityId > 0 &&
    Number.isFinite(caloriesConsumed) &&
    caloriesConsumed >= 0
      ? { date: day.date, activityId, caloriesConsumed }
      : null
  const strengthVolume = /^(\d+(?:\.\d+)?)\s*(kg|lbs?)$/i.exec(body.strengthvolume ?? '')
  const totalSets = Number(body.strengthsets)
  const totalReps = Number(body.strengthreps)
  const exercises = exerciseValues
    .map(parseStrengthExercise)
    .filter((exercise): exercise is StrengthExercise => exercise != null)
  const hasStrengthData =
    strengthVolume != null ||
    (Number.isSafeInteger(totalSets) && totalSets > 0) ||
    (Number.isSafeInteger(totalReps) && totalReps >= 0) ||
    exercises.length > 0
  const strength =
    hasStrengthData && Number.isSafeInteger(activityId) && activityId > 0
      ? {
          date: day.date,
          activityId,
          volumeKg: strengthVolume ? massKg(Number(strengthVolume[1]), strengthVolume[2]) : null,
          totalSets: Number.isSafeInteger(totalSets) && totalSets > 0 ? totalSets : null,
          totalReps: Number.isSafeInteger(totalReps) && totalReps >= 0 ? totalReps : null,
          exercises,
        }
      : null
  const skipTraining = ['true', '1', 'yes'].includes(body.skiptraining?.toLowerCase() ?? '')
  const trainingExclusion =
    skipTraining && Number.isSafeInteger(activityId) && activityId > 0
      ? { date: day.date, activityId }
      : null
  const sauna = parseManualSauna(body)
  return { day, fueling, strength, sauna, trainingExclusion }
}
