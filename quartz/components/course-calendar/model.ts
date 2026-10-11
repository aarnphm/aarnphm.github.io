import type { CalendarEvent } from '../controls/calendar'
import { isoDateFromLocal, parseIsoDate } from '../controls/dates'

/** What the build writes into the page for the calendar to lay out in the browser. */
export interface CourseCalendarUnit {
  n: number
  title: string
  part?: string
  sessions?: string
  href?: string
  state: 'new' | 'notes' | 'deck'
  read?: string
  problems?: string
  keyDates: string[]
  cards: number
}

export interface CourseCalendarPset {
  id: string
  label: string
  href?: string
  /** Unit the problem set is due at. */
  unit: number
  solutionHref?: string
}

export interface CourseCalendarData {
  number: string
  unitKind: 'lecture' | 'week' | 'session'
  total: number
  /** Lecture meetings per week and minutes per meeting; `source` says where they came from. */
  cadence: { sessions: number; minutes: number; source: 'syllabus' | 'default' }
  units: CourseCalendarUnit[]
  psets: CourseCalendarPset[]
  undated: string[]
  /** The course's study log page, where read and problems dates are kept. */
  logHref?: string
  /** `courses/<dir>/`, the flashcard state prefix. */
  prefix: string
  cardIds: string[]
}

export type CourseEventStatus = 'read' | 'problems' | 'planned' | 'due' | 'solved'

export interface CourseEvent extends CalendarEvent {
  status: CourseEventStatus
  unit?: CourseCalendarUnit
  pset?: CourseCalendarPset
}

// MIT meets on fixed weekdays: two sessions are Tuesday and Thursday, three are MWF.
const MEETING_DAYS: Record<number, number[]> = {
  1: [3],
  2: [2, 4],
  3: [1, 3, 5],
  4: [1, 2, 3, 4],
  5: [1, 2, 3, 4, 5],
  6: [1, 2, 3, 4, 5, 6],
  7: [0, 1, 2, 3, 4, 5, 6],
}

const localDate = (date: string) => {
  const parts = parseIsoDate(date)!
  return new Date(parts.year, parts.month - 1, parts.day)
}

const nextDay = (date: string) => {
  const day = localDate(date)
  day.setDate(day.getDate() + 1)
  return isoDateFromLocal(day)
}

/** Meeting days on or after `from`, one per meeting, or one per week for week-long units. */
function* meetings(from: string, sessions: number, weekly: boolean): Generator<string> {
  const days = weekly ? [1] : MEETING_DAYS[sessions]
  for (let date = from; ; date = nextDay(date)) {
    if (days.includes(localDate(date).getDay())) yield date
  }
}

export const pad = (n: number) => String(n).padStart(2, '0')

/**
 * Read units sit on their `study.read` day. Unread units take the course's meeting days in
 * order from today, or from the day after the latest read when that is later, so the plan
 * always starts at the next free meeting. A problem set falls on the day of its unit.
 */
export function courseEvents(data: CourseCalendarData, today: string): CourseEvent[] {
  const weekly = data.unitKind === 'week'
  const minutes = weekly ? data.cadence.sessions * data.cadence.minutes : data.cadence.minutes
  const lastRead = data.units.reduce(
    (last, unit) => (unit.read && unit.read > last ? unit.read : last),
    '',
  )
  const start = lastRead >= today ? nextDay(lastRead) : today
  const slots = meetings(start, data.cadence.sessions, weekly)
  const events: CourseEvent[] = []
  const unitDay = new Map<number, { date: string; planned: boolean }>()
  for (const unit of data.units) {
    const planned = !unit.read
    const date = unit.read ?? slots.next().value!
    unitDay.set(unit.n, { date, planned })
    events.push({
      id: `unit-${unit.n}`,
      date,
      title: `${pad(unit.n)} ${unit.title}`,
      kind: planned ? 'planned' : 'read',
      tone: 'secondary',
      planned,
      minutes,
      href: unit.href,
      status: planned ? 'planned' : 'read',
      unit,
    })
    if (unit.problems)
      events.push({
        id: `problems-${unit.n}`,
        date: unit.problems,
        title: `${pad(unit.n)} problems`,
        kind: 'problems',
        tone: 'iris',
        href: unit.href,
        status: 'problems',
        unit,
      })
  }
  for (const pset of data.psets) {
    const day = unitDay.get(pset.unit)
    if (!day) continue
    const solved = Boolean(pset.solutionHref)
    events.push({
      id: `pset-${pset.id}`,
      date: day.date,
      title: pset.label,
      kind: solved ? 'solved' : 'due',
      tone: 'love',
      planned: day.planned && !solved,
      href: pset.solutionHref ?? pset.href,
      status: solved ? 'solved' : 'due',
      pset,
    })
  }
  return events.sort((a, b) => a.date.localeCompare(b.date))
}

/** Cards due per day from now on; overdue cards count on today. */
export function dueByDay(
  rows: { cardId: string; due: number }[],
  cardIds: Set<string>,
  today: string,
): Map<string, number> {
  const counts = new Map<string, number>()
  for (const row of rows) {
    if (!cardIds.has(row.cardId)) continue
    const day = isoDateFromLocal(new Date(row.due))
    const key = day < today ? today : day
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return counts
}
