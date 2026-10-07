import fs from 'node:fs/promises'
import { dirname } from 'node:path'
import { joinSegments, QUARTZ } from './path'
import { parseTrainingPeaksCalendar, type TrainingPeaksCalendar } from './trainingpeaks-calendar'
import { isRecord } from './type-guards'

export const TRAININGPEAKS_CALENDAR_CACHE = joinSegments(
  QUARTZ,
  '.quartz-cache',
  'trainingpeaks',
  'calendar.json',
)

interface TrainingPeaksCalendarCache {
  athleteId: number
  calendar: TrainingPeaksCalendar
}

export async function readTrainingPeaksCalendarCache(
  path = TRAININGPEAKS_CALENDAR_CACHE,
): Promise<TrainingPeaksCalendarCache | null> {
  let value: unknown
  try {
    value = JSON.parse(await fs.readFile(path, 'utf8'))
  } catch (error) {
    if (isRecord(error) && error.code === 'ENOENT') return null
    throw new Error('TrainingPeaks calendar cache could not be read; retained the existing file', {
      cause: error,
    })
  }
  const calendar = parseTrainingPeaksCalendar(value)
  if (
    !calendar ||
    !isRecord(value) ||
    typeof value.athleteId !== 'number' ||
    !Number.isSafeInteger(value.athleteId) ||
    value.athleteId <= 0
  )
    throw new Error(
      'TrainingPeaks calendar cache has an invalid schema; retained the existing file',
    )
  return { athleteId: value.athleteId, calendar }
}

export async function readTrainingPeaksCalendar(): Promise<TrainingPeaksCalendar | null> {
  try {
    return (await readTrainingPeaksCalendarCache())?.calendar ?? null
  } catch (error) {
    console.warn(
      `[trainingpeaks-calendar] ${error instanceof Error ? error.message : 'Could not read local cache'}`,
    )
    return null
  }
}

export async function writeTrainingPeaksCalendarCache(
  cache: TrainingPeaksCalendarCache,
  path = TRAININGPEAKS_CALENDAR_CACHE,
): Promise<void> {
  await fs.mkdir(dirname(path), { recursive: true, mode: 0o700 })
  const temporary = `${path}.tmp-${process.pid}-${Date.now()}`
  try {
    await fs.writeFile(
      temporary,
      `${JSON.stringify({ ...cache.calendar, athleteId: cache.athleteId })}\n`,
      { mode: 0o600, flag: 'wx' },
    )
    await fs.rename(temporary, path)
  } finally {
    await fs.rm(temporary, { force: true })
  }
}
