export type TrailHoliday = 'thanksgiving' | 'halloween' | 'christmas' | 'easter'
export type HolidaySelection = TrailHoliday | 'auto' | 'none'

export const holidayLabels: Record<HolidaySelection, string> = {
  auto: 'automatic',
  none: 'off',
  thanksgiving: 'Thanksgiving',
  halloween: 'Halloween',
  christmas: 'Christmas',
  easter: 'Easter',
}

const dayMs = 86400000
const ordinal = (year: number, month: number, day: number) => Date.UTC(year, month, day) / dayMs

function nthWeekday(year: number, month: number, weekday: number, occurrence: number) {
  const first = new Date(Date.UTC(year, month, 1)).getUTCDay()
  return 1 + ((weekday - first + 7) % 7) + (occurrence - 1) * 7
}

function easter(year: number) {
  // Oudin's Gregorian computation, US Naval Observatory. Division truncates to integers.
  const century = Math.trunc(year / 100)
  const cycle = year % 19
  const correction = Math.trunc((century - 17) / 25)
  let moon =
    century - Math.trunc(century / 4) - Math.trunc((century - correction) / 3) + 19 * cycle + 15
  moon %= 30
  const adjustment = Math.trunc(moon / 28)
  moon -=
    adjustment * (1 - adjustment * Math.trunc(29 / (moon + 1)) * Math.trunc((21 - cycle) / 11))
  const weekday = (year + Math.trunc(year / 4) + moon + 2 - century + Math.trunc(century / 4)) % 7
  const offset = moon - weekday
  const month = 3 + Math.trunc((offset + 40) / 44)
  return { month: month - 1, day: offset + 28 - 31 * Math.trunc(month / 4) }
}

export function visitorHoliday(date: Date, country: string | null): TrailHoliday | null {
  const year = date.getFullYear()
  if (!Number.isFinite(date.getTime()) || year < 1583) return null
  const today = ordinal(year, date.getMonth(), date.getDate())
  const during = (month: number, day: number, before: number, after: number) => {
    const offset = today - ordinal(year, month, day)
    return offset >= -before && offset <= after
  }
  const us = country === 'US'
  if (during(us ? 10 : 9, nthWeekday(year, us ? 10 : 9, us ? 4 : 1, us ? 4 : 2), 3, 1))
    return 'thanksgiving'
  if (during(9, 31, 3, 1)) return 'halloween'
  if (during(11, 25, 5, 1)) return 'christmas'
  const spring = easter(year)
  return during(spring.month, spring.day, 2, 1) ? 'easter' : null
}

type HolidayEvents = {
  'visitor-holiday:request': CustomEvent<void>
  'visitor-holiday:change': CustomEvent<TrailHoliday | null>
  'trail:holiday': CustomEvent<TrailHoliday | null>
}

declare global {
  interface CustomEventMap extends HolidayEvents {}
  interface DocumentEventMap extends HolidayEvents {}
  interface HTMLElementEventMap {
    'trail:holiday': CustomEvent<TrailHoliday | null>
  }
}

function readHoliday(value: unknown): TrailHoliday | null {
  return value === 'thanksgiving' ||
    value === 'halloween' ||
    value === 'christmas' ||
    value === 'easter'
    ? value
    : null
}

export function createVisitorHolidaySource(target: Document, signal: AbortSignal) {
  let latest: TrailHoliday | null = null
  const dispatch = () =>
    target.dispatchEvent(new CustomEvent('visitor-holiday:change', { detail: latest }))
  target.addEventListener('visitor-holiday:request', dispatch, { signal })
  signal.addEventListener(
    'abort',
    () => {
      latest = null
      dispatch()
    },
    { once: true },
  )
  return {
    publish(country: string | null, now = Date.now()) {
      if (signal.aborted) return
      const next = visitorHoliday(new Date(now), country)
      if (next === latest) return
      latest = next
      dispatch()
    },
  }
}

export function subscribeVisitorHoliday(
  target: Document,
  receive: (holiday: TrailHoliday | null) => void,
  signal: AbortSignal,
) {
  target.addEventListener(
    'visitor-holiday:change',
    event => {
      if ('detail' in event) receive(readHoliday(event.detail))
    },
    { signal },
  )
  target.dispatchEvent(new CustomEvent<void>('visitor-holiday:request'))
}
