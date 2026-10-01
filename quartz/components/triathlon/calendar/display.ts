import type { Locale } from '../../../util/triathlon-presentation'

export type CalendarDatePart = 'full' | 'month' | 'weekday' | 'day' | 'short'

const calendarLocale = (locale: Locale): string => (locale === 'fr' ? 'fr-CA' : 'en-CA')

export const calendarDateLabel = (
  start: string,
  end: string | null,
  part: CalendarDatePart,
  locale: Locale = 'en',
): string => {
  const first = new Date(`${start}T12:00:00Z`)
  const last = end ? new Date(`${end}T12:00:00Z`) : first
  if (part === 'day') {
    const day = start.slice(8)
    return end && end !== start ? `${day}–${end.slice(8)}` : day
  }
  const options: Intl.DateTimeFormatOptions =
    part === 'month'
      ? { month: 'short' }
      : part === 'weekday'
        ? { weekday: 'short' }
        : part === 'short'
          ? { weekday: 'short', day: 'numeric' }
          : { day: 'numeric', month: 'long', year: 'numeric' }
  const format = new Intl.DateTimeFormat(calendarLocale(locale), { ...options, timeZone: 'UTC' })
  if (part === 'full')
    return first.getTime() === last.getTime()
      ? format.format(first)
      : format.formatRange(first, last)
  const firstLabel = format.format(first)
  const lastLabel = format.format(last)
  return firstLabel === lastLabel ? firstLabel : `${firstLabel}–${lastLabel}`
}

export const calendarMonthLabel = (
  year: number,
  month: number,
  locale: Locale = 'en',
  short = false,
): string =>
  new Intl.DateTimeFormat(calendarLocale(locale), {
    month: short ? 'short' : 'long',
    timeZone: 'UTC',
  }).format(new Date(Date.UTC(year, month, 1)))

export const calendarWeekdayLabel = (day: number, locale: Locale = 'en'): string =>
  new Intl.DateTimeFormat(calendarLocale(locale), { weekday: 'long', timeZone: 'UTC' }).format(
    new Date(Date.UTC(2024, 0, day + 1)),
  )

export const calendarToday = (): string =>
  new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Toronto' }).format(new Date())
