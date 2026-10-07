import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarMetrics,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import type { Locale } from '../../../util/triathlon-presentation'
import { isTrainingPeaksCalendarDay } from '../../../util/trainingpeaks-calendar'

const DAY_MS = 86_400_000

export const trainingDate = (value: string): Date | null => {
  return isTrainingPeaksCalendarDay(value) ? new Date(`${value}T12:00:00Z`) : null
}

export const trainingAddDays = (value: string, days: number): string => {
  const date = trainingDate(value)
  return date ? new Date(date.getTime() + days * DAY_MS).toISOString().slice(0, 10) : value
}

export const trainingWeekStart = (value: string): string => {
  const date = trainingDate(value)
  return date ? trainingAddDays(value, -((date.getUTCDay() + 6) % 7)) : value
}

export const trainingWeekDates = (start: string): string[] =>
  Array.from({ length: 7 }, (_, day) => trainingAddDays(start, day))

export const trainingMonthStart = (value: string): string =>
  trainingDate(value) ? `${value.slice(0, 7)}-01` : value

export const trainingAddMonths = (value: string, months: number): string => {
  const date = trainingDate(trainingMonthStart(value))
  if (!date) return value
  date.setUTCMonth(date.getUTCMonth() + months)
  return date.toISOString().slice(0, 10)
}

export const trainingMonthDates = (value: string): string[] => {
  const start = trainingWeekStart(trainingMonthStart(value))
  const end = trainingAddDays(
    trainingWeekStart(trainingAddDays(trainingAddMonths(value, 1), -1)),
    6,
  )
  const first = trainingDate(start)
  const last = trainingDate(end)
  if (!first || !last) return []
  const days = Math.round((last.getTime() - first.getTime()) / DAY_MS) + 1
  return Array.from({ length: days }, (_, day) => trainingAddDays(start, day))
}

export const trainingWeekLabel = (start: string, locale: Locale): string => {
  const first = trainingDate(start)
  const last = trainingDate(trainingAddDays(start, 6))
  if (!first || !last) return start
  return new Intl.DateTimeFormat(locale === 'fr' ? 'fr-CA' : 'en-CA', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).formatRange(first, last)
}

export const trainingDayCovered = (calendar: TrainingPeaksCalendar, date: string): boolean =>
  calendar.coverage.some(range => date >= range.since && date <= range.until)

export const trainingWorkouts = (
  calendar: TrainingPeaksCalendar | null,
  start: string,
  end: string,
): TrainingPeaksCalendarWorkout[] => {
  return (calendar?.workouts ?? [])
    .filter(workout => workout.date >= start && workout.date <= end)
    .sort(
      (first, second) =>
        first.date.localeCompare(second.date) ||
        (first.order ?? Number.MAX_SAFE_INTEGER) - (second.order ?? Number.MAX_SAFE_INTEGER) ||
        (first.startTimePlanned ?? first.startTime ?? '').localeCompare(
          second.startTimePlanned ?? second.startTime ?? '',
        ) ||
        first.id.localeCompare(second.id),
    )
}

export const trainingTotal = (
  workouts: readonly TrainingPeaksCalendarWorkout[],
  source: 'planned' | 'actual',
  metric: keyof TrainingPeaksCalendarMetrics,
): number | null => {
  const values = workouts.flatMap(workout => {
    const value = workout[source][metric]
    return value === null ? [] : [value]
  })
  return values.length > 0 ? values.reduce((total, value) => total + value, 0) : null
}

export const trainingDuration = (seconds: number | null): string => {
  if (seconds === null) return '—'
  const minutes = Math.round(seconds / 60)
  if (seconds > 0 && minutes === 0) return '<1 min'
  const hours = Math.floor(minutes / 60)
  const remainder = minutes % 60
  return hours ? `${hours} h${remainder ? ` ${remainder} min` : ''}` : `${minutes} min`
}
