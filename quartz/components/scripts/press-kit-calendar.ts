export interface CalendarRace {
  id: string
  title: string
  location: string
  format: string
  month: string
  day: string
  weekday: string
  pending: string
  status: string
  color: string
  onColor: string
  schedule: { label: string; race: boolean }[]
}

export interface CalendarCardData {
  title: string
  period: string
  url: string
  empty: string
  races: CalendarRace[]
}

const text = (element: Element, selector: string): string =>
  element.querySelector(selector)?.textContent?.trim() ?? ''

export function calendarCardData(calendar: HTMLElement): CalendarCardData | null {
  const period = calendar.dataset.calendarYear
  if (!period) return null
  const url = new URL('/triathlon/calendar', location.href)
  url.hash = period
  // Every calendar view owns the same list, including its localized date and schedule labels.
  const races = Array.from(
    calendar.querySelectorAll<HTMLElement>('.tri-calendar-list > .tri-calendar-event'),
    row => {
      const style = getComputedStyle(row)
      return {
        id: row.dataset.calendarCard ?? row.id,
        title: text(row, '.tri-calendar-event-heading > button'),
        location: text(row, '.tri-calendar-event-meta > span'),
        format: text(row, '.tri-calendar-format'),
        month: text(row, '.tri-calendar-date-month'),
        day: text(row, '.tri-calendar-date-day'),
        weekday: text(row, '.tri-calendar-date-weekday'),
        pending: text(row, '.tri-calendar-pending'),
        status: text(row, '.tri-calendar-next:not([hidden]), .tri-calendar-participated'),
        color: style.getPropertyValue('--calendar-event').trim(),
        onColor: style.getPropertyValue('--calendar-event-on').trim(),
        schedule: Array.from(
          row.querySelectorAll<HTMLElement>('.tri-calendar-schedule-day'),
          day => ({
            label: [text(day, 'time'), text(day, '.tri-calendar-start')].filter(Boolean).join(' '),
            race: day.dataset.race === 'true',
          }),
        ),
      }
    },
  )
  return {
    title: `${text(calendar, '.tri-calendar-title-group :is(h1, h2)')} · ${period}`,
    period,
    url: url.href,
    empty: text(calendar, '[data-i18n="No events planned."]'),
    races,
  }
}
