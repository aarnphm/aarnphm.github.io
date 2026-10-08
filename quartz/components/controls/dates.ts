// Local calendar days as `YYYY-MM-DD` strings. String comparison orders them.

export interface DateParts {
  year: number
  month: number
  day: number
}

export interface MonthParts {
  year: number
  month: number
}

const pad2 = (value: number): string => String(value).padStart(2, '0')

export const isoDate = (year: number, month: number, day: number): string =>
  `${year}-${pad2(month)}-${pad2(day)}`

export const isoDateFromLocal = (date: Date): string =>
  isoDate(date.getFullYear(), date.getMonth() + 1, date.getDate())

export const parseIsoDate = (value: string | undefined): DateParts | null => {
  if (!value) return null
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const date = new Date(year, month - 1, day)
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day)
    return null
  return { year, month, day }
}

export const isoMonth = (parts: MonthParts): string => `${parts.year}-${pad2(parts.month)}`

export const parseIsoMonth = (value: string | undefined): MonthParts | null => {
  if (!value) return null
  const match = /^(\d{4})-(\d{2})$/.exec(value)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  if (month < 1 || month > 12) return null
  return { year, month }
}

export const monthOf = (parts: DateParts): MonthParts => ({ year: parts.year, month: parts.month })

export const addMonths = (parts: MonthParts, delta: number): MonthParts => {
  const date = new Date(parts.year, parts.month - 1 + delta, 1)
  return { year: date.getFullYear(), month: date.getMonth() + 1 }
}

export const todayParts = (): DateParts => {
  const date = new Date()
  return { year: date.getFullYear(), month: date.getMonth() + 1, day: date.getDate() }
}

export const clampIsoDate = (
  value: string,
  min: string | undefined,
  max: string | undefined,
): string => {
  if (min && value < min) return min
  if (max && value > max) return max
  return value
}

/** Moves `value` by `offset` days inside [min, max] and names the month that holds the result. */
export const shiftIsoDate = (
  value: string | undefined,
  offset: number,
  min: string | undefined,
  max: string | undefined,
): { date: string; viewMonth: string } | null => {
  const current = parseIsoDate(value)
  if (!current) return null
  const moved = new Date(current.year, current.month - 1, current.day + offset)
  const date = clampIsoDate(isoDateFromLocal(moved), min, max)
  const parts = parseIsoDate(date)
  return parts ? { date, viewMonth: isoMonth(monthOf(parts)) } : null
}

/** The 6 × 7 days that show `view`, starting on the Sunday on or before its first day. */
export const monthGridDates = (view: MonthParts): string[] => {
  const first = new Date(view.year, view.month - 1, 1)
  return Array.from({ length: 42 }, (_, index) =>
    isoDateFromLocal(new Date(view.year, view.month - 1, 1 - first.getDay() + index)),
  )
}
