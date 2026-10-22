import { render } from 'preact'
import { useEffect, useMemo, useState } from 'preact/hooks'
import {
  Calendar,
  englishCalendarLabels,
  type CalendarEvent,
  type CalendarLabels,
} from '../controls/calendar'
import { frenchDateLabels } from '../controls/date-picker'
import { isoDateFromLocal, parseIsoDate } from '../controls/dates'

/** What the build writes into the page: the dated notes, and where the hand-written schedule is. */
export interface FrCalendarData {
  notes: { date: string; kind: 'lessons' | 'practices'; href: string; episodes: number[] }[]
  horaire: string
  /** `horaire.json` read at build, so the first paint already has the schedule. */
  schedule?: Horaire
  /** `fr/`, the flashcard state prefix. */
  prefix: string
  cardIds: string[]
}

/** `content/fr/horaire.json`, as written by hand. */
export interface Horaire {
  fuseau?: string
  recurrences?: {
    quoi: string
    jour: string
    de: string
    à: string
    depuis?: string
    lien?: string
  }[]
  blocs?: { date: string; quoi: string; de: string; à: string; lien?: string }[]
  sauf?: string[]
}

type FrEvent = CalendarEvent & { source: 'note' | 'plan'; episodes?: number[] }

const DAYS = ['dim', 'lun', 'mar', 'mer', 'jeu', 'ven', 'sam']

export const frLabels: CalendarLabels = {
  ...englishCalendarLabels,
  locale: 'fr-CA',
  picker: frenchDateLabels,
  previous: 'mois précédent',
  next: 'mois suivant',
  today: "aujourd'hui",
  jump: 'choisir le mois',
  none: 'rien',
}

const localDate = (date: string) => {
  const parts = parseIsoDate(date)!
  return new Date(parts.year, parts.month - 1, parts.day)
}

const minutesBetween = (from: string, to: string) => {
  const [h1, m1] = from.split(':').map(Number)
  const [h2, m2] = to.split(':').map(Number)
  return (h2 ?? 0) * 60 + (m2 ?? 0) - ((h1 ?? 0) * 60 + (m1 ?? 0))
}

/** `ép. 11–13` for a run of episodes, `ép. 11, 14` for a gap. */
export function episodeWords(episodes: number[]): string {
  if (episodes.length === 0) return ''
  const first = episodes[0]
  const last = episodes[episodes.length - 1]
  if (episodes.length === 1) return `ép. ${first}`
  return last - first === episodes.length - 1
    ? `ép. ${first}–${last}`
    : `ép. ${episodes.join(', ')}`
}

const KIND_WORDS = { lessons: 'leçon', practices: 'pratique' } as const
const LINK_KIND: Record<string, 'lessons' | 'practices'> = {
  'fr/lessons': 'lessons',
  'fr/practices': 'practices',
}

/**
 * Dated notes become chips on their day. Each recurrence of `horaire.json` is unrolled from its
 * start to a year past `today`; a recurrence whose `lien` has a note that day is the note
 * instead, so a lesson that happened replaces the class that planned it.
 */
export function frEvents(data: FrCalendarData, horaire: Horaire | null, today: string): FrEvent[] {
  const events: FrEvent[] = data.notes.map(note => ({
    id: `${note.kind}-${note.date}`,
    date: note.date,
    title: [KIND_WORDS[note.kind], episodeWords(note.episodes)].filter(Boolean).join(' · '),
    kind: KIND_WORDS[note.kind],
    tone: note.kind === 'lessons' ? 'secondary' : 'tertiary',
    href: note.href,
    source: 'note',
    episodes: note.episodes,
  }))
  if (!horaire) return events.sort((a, b) => a.date.localeCompare(b.date))

  const noted = new Set(data.notes.map(note => `${note.kind}|${note.date}`))
  const skipped = new Set(horaire.sauf ?? [])
  const plan = (
    date: string,
    what: string,
    from: string,
    to: string,
    link?: string,
    reminder?: boolean,
  ) => {
    // a skipped day cancels the class, not the daily reminders
    if (skipped.has(date) && !reminder) return
    const kind = link ? LINK_KIND[link] : undefined
    if (kind && noted.has(`${kind}|${date}`)) return
    events.push({
      id: `plan-${what}-${date}`,
      date,
      title: what,
      time: from,
      kind: 'prévu',
      tone: kind === 'practices' ? 'tertiary' : 'secondary',
      planned: true,
      thin: reminder,
      minutes: reminder ? undefined : minutesBetween(from, to),
      source: 'plan',
    })
  }
  const end = localDate(today)
  end.setFullYear(end.getFullYear() + 1)
  for (const rule of horaire.recurrences ?? []) {
    const daily = rule.jour === 'tous'
    const day = daily ? 0 : DAYS.indexOf(rule.jour)
    if (day < 0 || !parseIsoDate(rule.depuis ?? today)) continue
    const at = localDate(rule.depuis ?? today)
    if (!daily) at.setDate(at.getDate() + ((day - at.getDay() + 7) % 7))
    for (; at <= end; at.setDate(at.getDate() + (daily ? 1 : 7)))
      plan(isoDateFromLocal(at), rule.quoi, rule.de, rule.à, rule.lien, daily)
  }
  for (const bloc of horaire.blocs ?? []) plan(bloc.date, bloc.quoi, bloc.de, bloc.à, bloc.lien)
  return events.sort(
    (a, b) => a.date.localeCompare(b.date) || (a.time ?? '').localeCompare(b.time ?? ''),
  )
}

interface StateResponse {
  login?: string | null
  states?: { cardId: string; due: number }[]
}

const Preview = ({ event }: { event: FrEvent }) => (
  <>
    <h4>{event.title}</h4>
    <dl class="g-kv">
      <dt>date</dt>
      <dd>{event.date}</dd>
      {event.time ? (
        <>
          <dt>heure</dt>
          <dd>
            {event.time}
            {event.minutes ? ` · ${event.minutes} min` : ''}
          </dd>
        </>
      ) : null}
      <dt>état</dt>
      <dd>{event.source === 'plan' ? 'prévu' : 'fait'}</dd>
    </dl>
  </>
)

/** Server-rendered with the schedule; the browser adds the due counts. */
export const FrCalendarView = ({
  id,
  data,
  signal,
}: {
  id: string
  data: FrCalendarData
  signal?: AbortSignal
}) => {
  const today = isoDateFromLocal(new Date())
  const [horaire, setHoraire] = useState<Horaire | null>(data.schedule ?? null)
  const [due, setDue] = useState<Map<string, number>>(new Map())
  const events = useMemo(() => frEvents(data, horaire, today), [data, horaire, today])

  useEffect(() => {
    if (!data.schedule)
      fetch(data.horaire, { signal })
        .then(res => (res.ok ? (res.json() as Promise<Horaire>) : null))
        .then(body => {
          if (!signal?.aborted && body) setHoraire(body)
        })
        .catch(() => undefined)
    if (data.cardIds.length === 0) return
    const ids = new Set(data.cardIds)
    fetch(`/api/flashcards/state?prefix=${encodeURIComponent(data.prefix)}`, { signal })
      .then(res => (res.ok ? (res.json() as Promise<StateResponse>) : null))
      .then(body => {
        if (signal?.aborted || !body?.login || !body.states) return
        const counts = new Map<string, number>()
        for (const row of body.states) {
          if (!ids.has(row.cardId)) continue
          const day = isoDateFromLocal(new Date(row.due))
          const key = day < today ? today : day
          counts.set(key, (counts.get(key) ?? 0) + 1)
        }
        setDue(counts)
      })
      .catch(() => undefined)
  }, [data])

  const start = events.some(event => event.date.slice(0, 7) === today.slice(0, 7))
    ? today.slice(0, 7)
    : (events.findLast(event => event.date < today)?.date.slice(0, 7) ?? today.slice(0, 7))

  return (
    <Calendar
      id={id}
      events={events}
      today={today}
      start={start}
      labels={frLabels}
      dayHeader={date => {
        const count = due.get(date)
        return count ? (
          <span class="course-calendar-due" title={`${count} à revoir`}>
            {count} à revoir
          </span>
        ) : null
      }}
      previewLabel={event => `${event.title}, ${event.date}`}
      renderPreview={event => <Preview event={event as FrEvent} />}
    />
  )
}

export const FR_CALENDAR_ID = 'fr-calendar-grid'

export function mountFrCalendar(root: HTMLElement, signal: AbortSignal): () => void {
  const source = root.querySelector<HTMLScriptElement>('script[data-fr-calendar-data]')
  const mount = root.querySelector<HTMLElement>('[data-fr-calendar-mount]')
  if (!source || !mount) return () => undefined
  const data = JSON.parse(source.textContent ?? 'null') as FrCalendarData | null
  if (!data) return () => undefined
  mount.replaceChildren()
  render(<FrCalendarView id={FR_CALENDAR_ID} data={data} signal={signal} />, mount)
  return () => render(null, mount)
}
