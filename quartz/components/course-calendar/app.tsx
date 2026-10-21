import { render } from 'preact'
import { useEffect, useMemo, useState } from 'preact/hooks'
import { Calendar, calendarDuration } from '../controls/calendar'
import { isoDateFromLocal } from '../controls/dates'
import {
  CourseCalendarData,
  CourseEvent,
  CourseEventStatus,
  courseEvents,
  dueByDay,
  pad,
} from './model'

const STATUS_WORDS: Record<CourseEventStatus, string> = {
  read: 'read',
  problems: 'problems done',
  planned: 'planned',
  due: 'problem set due',
  solved: 'problem set solved',
}

const LEGEND: { status: CourseEventStatus; tone: string; planned?: boolean; text: string }[] = [
  { status: 'read', tone: 'secondary', text: 'read' },
  { status: 'planned', tone: 'secondary', planned: true, text: 'planned' },
  { status: 'problems', tone: 'iris', text: 'problems' },
  { status: 'due', tone: 'love', planned: true, text: 'problem set' },
]

const Preview = ({ event, data }: { event: CourseEvent; data: CourseCalendarData }) => {
  const { unit, pset } = event
  return (
    <>
      <h4>{unit ? unit.title : event.title}</h4>
      <dl class="g-kv">
        <dt>{unit ? data.unitKind : 'set'}</dt>
        <dd>
          {unit ? `${pad(unit.n)} of ${data.total}` : `due at ${data.unitKind} ${pad(pset!.unit)}`}
        </dd>
        {unit?.part ? (
          <>
            <dt>part</dt>
            <dd>{unit.part}</dd>
          </>
        ) : null}
        {unit?.sessions ? (
          <>
            <dt>sessions</dt>
            <dd>{unit.sessions}</dd>
          </>
        ) : null}
        <dt>status</dt>
        <dd>
          {STATUS_WORDS[event.status]} {event.date}
        </dd>
        {event.minutes ? (
          <>
            <dt>length</dt>
            <dd>
              {calendarDuration(event.minutes)}
              {data.cadence.source === 'syllabus' ? ' per the ocw syllabus' : ' assumed'}
            </dd>
          </>
        ) : null}
        {unit && unit.keyDates.length > 0 ? (
          <>
            <dt>key dates</dt>
            <dd>{unit.keyDates.join(' · ')}</dd>
          </>
        ) : null}
        {unit ? (
          <>
            <dt>notes</dt>
            <dd>
              {unit.state === 'new'
                ? 'none yet'
                : unit.state === 'deck'
                  ? `${unit.cards} cards`
                  : 'written'}
            </dd>
          </>
        ) : null}
      </dl>
    </>
  )
}

interface StateResponse {
  login?: string | null
  states?: { cardId: string; due: number }[]
}

/** Server-rendered at build time with that day's plan; the browser re-renders it for its today. */
export const CourseCalendarView = ({
  id,
  data,
  signal,
}: {
  id: string
  data: CourseCalendarData
  signal?: AbortSignal
}) => {
  const today = isoDateFromLocal(new Date())
  const events = useMemo(() => courseEvents(data, today), [data, today])
  const [due, setDue] = useState<Map<string, number>>(new Map())

  // The owner's FSRS schedule; signed out, the day headers stay empty.
  useEffect(() => {
    if (data.cardIds.length === 0) return
    const ids = new Set(data.cardIds)
    fetch(`/api/flashcards/state?prefix=${encodeURIComponent(data.prefix)}`, { signal })
      .then(res => (res.ok ? (res.json() as Promise<StateResponse>) : null))
      .then(body => {
        if (!signal?.aborted && body?.login && body.states)
          setDue(dueByDay(body.states, ids, today))
      })
      .catch(() => undefined)
  }, [data])

  const kindPlural = `${data.unitKind}s`
  // a course read in the past opens on its last month; one not started opens on today
  const start = events.some(event => event.date.slice(0, 7) === today.slice(0, 7))
    ? today.slice(0, 7)
    : (events.findLast(event => event.date < today)?.date.slice(0, 7) ?? today.slice(0, 7))

  return (
    <>
      <Calendar
        id={id}
        events={events}
        today={today}
        start={start}
        summary={(_, shown) => {
          const units = shown.filter(event => event.unit && event.status !== 'problems')
          const read = units.filter(event => event.status === 'read').length
          const minutes = units.reduce((sum, event) => sum + (event.minutes ?? 0), 0)
          return (
            <>
              <strong>{read}</strong> read · <strong>{units.length - read}</strong> planned{' '}
              {kindPlural} · <strong>{calendarDuration(minutes)}</strong>
            </>
          )
        }}
        dayHeader={date => {
          const count = due.get(date)
          return count ? (
            <span
              class="course-calendar-due"
              title={`${count} ${count === 1 ? 'card' : 'cards'} due`}
            >
              {count} due
            </span>
          ) : null
        }}
        previewLabel={event => `${event.title}, ${STATUS_WORDS[event.status]}`}
        renderPreview={event => <Preview event={event} data={data} />}
      />
      <div class="course-calendar-foot">
        <ul class="course-calendar-legend" aria-label="event kinds">
          {LEGEND.map(item => (
            <li key={item.status}>
              <span
                class="g-event-mark"
                data-tone={item.tone}
                data-planned={item.planned ? 'true' : undefined}
                aria-hidden="true"
              />
              {item.text}
            </li>
          ))}
        </ul>
        <p>
          read and problems dates come from{' '}
          {data.logHref ? (
            <a class="internal" href={data.logHref}>
              the study log
            </a>
          ) : (
            'the study log, which this course does not have yet'
          )}
          . unread {kindPlural} fill the next meetings from today, {data.cadence.sessions} a week,{' '}
          {calendarDuration(data.cadence.minutes)} each
          {data.cadence.source === 'syllabus' ? ' (ocw syllabus)' : ' (assumed)'}.
          {data.undated.length > 0 ? ` undated: ${data.undated.join(' · ')}.` : ''}
        </p>
      </div>
    </>
  )
}

export const COURSE_CALENDAR_ID = 'course-calendar-grid'

export function mountCourseCalendar(root: HTMLElement, signal: AbortSignal): () => void {
  const source = root.querySelector<HTMLScriptElement>('script[data-course-calendar-data]')
  const mount = root.querySelector<HTMLElement>('[data-course-calendar-mount]')
  if (!source || !mount) return () => undefined
  const data = JSON.parse(source.textContent ?? 'null') as CourseCalendarData | null
  if (!data) return () => undefined
  // the build's markup goes in the same task, so the swap never paints an empty mount
  mount.replaceChildren()
  render(<CourseCalendarView id={COURSE_CALENDAR_ID} data={data} signal={signal} />, mount)
  return () => render(null, mount)
}

export { FR_CALENDAR_ID, FrCalendarView, mountFrCalendar } from './fr'
