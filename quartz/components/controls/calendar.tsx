import type { ComponentChildren } from 'preact'
import { useLayoutEffect, useRef, useState } from 'preact/hooks'
import {
  buildDatePicker,
  CALENDAR_ICON_PATH,
  englishDateLabels,
  type DatePickerLabels,
} from './date-picker'
import { addMonths, isoMonth, monthGridDates, parseIsoMonth } from './dates'
import { MonthGrid } from './month-grid'
import { useAnchoredPreview } from './use-anchored-preview'

export interface CalendarEvent {
  id: string
  date: string
  title: string
  /** Word for the chip label and the preview, such as "read" or "due". */
  kind: string
  /** `secondary`, `iris`, `love`, `rose`, `foam`, `gold`, or a sport (`swim`, `bike`, `run`). */
  tone?: string
  /** A planned event draws its chip as an outline. */
  planned?: boolean
  minutes?: number
  /** A reminder: a short one-line chip with no duration. */
  thin?: boolean
  /** Clock time that leads the chip, such as `18:30`. */
  time?: string
  /** A chip with a page opens it; a chip without one is a button that only previews. */
  href?: string
}

export const calendarDuration = (minutes: number) =>
  `${Math.floor(minutes / 60)}:${String(Math.round(minutes % 60)).padStart(2, '0')}`

export interface CalendarLabels {
  locale: string
  picker: DatePickerLabels
  previous: string
  next: string
  today: string
  jump: string
  none: string
}

export const englishCalendarLabels: CalendarLabels = {
  locale: 'en',
  picker: englishDateLabels,
  previous: 'previous month',
  next: 'next month',
  today: 'today',
  jump: 'choose month',
  none: 'no events',
}

const weekday = (locale: string) => (date: string) =>
  new Date(`${date}T00:00:00`).toLocaleDateString(locale, { weekday: 'short' }).toLowerCase()
const dayName = (locale: string) => (date: string) =>
  new Date(`${date}T00:00:00`)
    .toLocaleDateString(locale, { weekday: 'long', month: 'long', day: 'numeric' })
    .toLowerCase()

const Arrow = ({ d }: { d: string }) => (
  <svg class="g-icon" viewBox="0 0 16 16" fill="none" aria-hidden="true" focusable="false">
    <path
      d={d}
      stroke="currentColor"
      stroke-width="1.6"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  </svg>
)

// The month jump is the shared date picker. Its trigger and panel shell are rendered here, so the
// build's HTML already has them; the picker only attaches its handlers and fills the open panel.
const MonthJump = ({
  id,
  month,
  onPick,
  labels,
}: {
  id: string
  month: string
  onPick: (month: string) => void
  labels: CalendarLabels
}) => {
  const host = useRef<HTMLDivElement>(null)
  const state = useRef({ month, onPick })
  state.current = { month, onPick }
  useLayoutEffect(() => {
    const picker = buildDatePicker({
      id,
      host: host.current!,
      label: labels.jump,
      labels: labels.picker,
      selected: () => `${state.current.month}-01`,
      onSelect: date => state.current.onPick(date.slice(0, 7)),
    })
    const unmount = picker.mount()
    return () => {
      picker.close()
      unmount()
    }
  }, [])
  return (
    <div class="g-calendar-jump">
      <div ref={host} class="g-datepicker">
        <button
          type="button"
          class="g-datepicker-trigger"
          aria-label={labels.jump}
          aria-haspopup="dialog"
          aria-expanded="false"
          aria-controls={id}
        >
          <span class="g-datepicker-text" aria-live="polite">
            {labels.picker.monthYear(`${month}-01`)}
          </span>
          <svg
            class="g-datepicker-icon"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
            focusable="false"
          >
            <path
              d={CALENDAR_ICON_PATH}
              stroke="currentColor"
              stroke-width="1.35"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
        <div
          id={id}
          class="g-datepicker-panel"
          tabIndex={-1}
          role="dialog"
          aria-label={`${labels.jump} · ${labels.picker.text('date picker')}`}
          popover="auto"
        />
      </div>
    </div>
  )
}

const Chip = ({
  event,
  describedBy,
  anchor,
}: {
  event: CalendarEvent
  describedBy?: string
  anchor: Record<string, unknown>
}) => {
  const props = {
    class: 'g-event',
    'data-tone': event.tone,
    'data-event-id': event.id,
    'data-planned': event.planned ? 'true' : undefined,
    'data-thin': event.thin ? 'true' : undefined,
    'aria-label': `${event.title}, ${event.kind}${event.minutes ? `, ${calendarDuration(event.minutes)}` : ''}`,
    'aria-describedby': describedBy,
    ...anchor,
  }
  const body = (
    <>
      <span class="g-event-mark" aria-hidden="true" />
      <span class="g-event-title">
        {event.time ? <span class="g-event-time">{event.time} </span> : null}
        {event.title}
      </span>
      {event.minutes ? <span class="g-event-meta">{calendarDuration(event.minutes)}</span> : null}
    </>
  )
  // the chip carries its own preview, so the site's link popover stays off
  return event.href ? (
    <a href={event.href} data-no-popover {...props}>
      {body}
    </a>
  ) : (
    <button type="button" {...props}>
      {body}
    </button>
  )
}

/**
 * A month of event chips with previous/today/next, a month jump, a summary and a preview per
 * chip: the design system's Calendar. The host owns what the summary, the day headers and the
 * preview say.
 */
export const Calendar = <T extends CalendarEvent>({
  id,
  events,
  today,
  start,
  summary,
  dayHeader,
  renderPreview,
  previewLabel,
  labels = englishCalendarLabels,
}: {
  id: string
  events: T[]
  today: string
  /** `YYYY-MM` shown first; the month of `today` when absent. */
  start?: string
  summary?: (month: string, shown: T[]) => ComponentChildren
  dayHeader?: (date: string) => ComponentChildren
  renderPreview: (event: T) => ComponentChildren
  previewLabel?: (event: T) => string
  labels?: CalendarLabels
}) => {
  const [month, setMonth] = useState(start ?? today.slice(0, 7))
  const view = parseIsoMonth(month) ?? parseIsoMonth(today.slice(0, 7))!
  const all = monthGridDates(view)
  // Drop a trailing week that holds only next-month days.
  const dates = all[35].slice(0, 7) === month ? all : all.slice(0, 35)
  const shown = events.filter(event => event.date.startsWith(month))
  const { preview, popoverRef, anchorProps, popoverProps, onKeyDown, dismissFrom, openFrom } =
    useAnchoredPreview<T>({ resetKeys: [month] })
  // The chip whose preview was open when the pointer went down. A tap focuses the chip, which
  // opens its preview before the click, so the click must toggle from the state before the tap.
  const openAtPress = useRef<HTMLElement | null | undefined>(undefined)
  const step = (n: number) => setMonth(isoMonth(addMonths(view, n)))

  return (
    <>
      <div class="g-calendar-bar">
        <div class="g-calendar-nav">
          <button
            type="button"
            class="g-icon-button"
            aria-label={labels.previous}
            onClick={() => step(-1)}
          >
            <Arrow d="M10 3.5 5.5 8l4.5 4.5" />
          </button>
          <button type="button" class="g-text-button" onClick={() => setMonth(today.slice(0, 7))}>
            {labels.today}
          </button>
          <button
            type="button"
            class="g-icon-button"
            aria-label={labels.next}
            onClick={() => step(1)}
          >
            <Arrow d="M6 3.5 10.5 8 6 12.5" />
          </button>
        </div>
        <MonthJump id={`${id}-jump`} month={month} onPick={setMonth} labels={labels} />
        {summary ? <p class="g-calendar-summary">{summary(month, shown)}</p> : null}
      </div>
      <MonthGrid
        id={id}
        dates={dates}
        month={month}
        today={today}
        weekdayLabel={weekday(labels.locale)}
        dayLabel={dayName(labels.locale)}
        dayHeader={dayHeader}
        onKeyDown={onKeyDown}
        onPointerDownCapture={() => {
          openAtPress.current = preview?.anchor ?? null
        }}
        onClickCapture={event => {
          if (!(event.target instanceof Element)) return
          const chip = event.target.closest<HTMLElement>('.g-event')
          const wasOpen = (openAtPress.current ?? preview?.anchor) === chip
          openAtPress.current = undefined
          if (!chip) return
          // a link chip opens its page; a button chip has only its preview, so it toggles
          const item = events.find(entry => entry.id === chip.dataset.eventId)
          if (chip instanceof HTMLButtonElement && item && !wasOpen) openFrom(item, chip)
          else dismissFrom(chip)
        }}
        renderDay={date => {
          const day = events.filter(event => event.date === date)
          return (
            <div class="g-calendar-events">
              {day.map(event => (
                <Chip
                  key={event.id}
                  event={event}
                  describedBy={preview?.item.id === event.id ? `${id}-preview` : undefined}
                  anchor={anchorProps(event)}
                />
              ))}
              {day.length === 0 && (
                <span class="g-calendar-empty" aria-label={labels.none}>
                  –
                </span>
              )}
            </div>
          )
        }}
      >
        <div
          ref={popoverRef}
          id={`${id}-preview`}
          class="g-calendar-pop"
          popover="manual"
          role="region"
          aria-label={
            preview ? (previewLabel?.(preview.item) ?? `${preview.item.title} preview`) : undefined
          }
          {...popoverProps}
        >
          {preview && renderPreview(preview.item)}
        </div>
      </MonthGrid>
    </>
  )
}
