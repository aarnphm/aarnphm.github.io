import { useState } from 'preact/hooks'
import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarActivityLink,
  TrainingPeaksCalendarPeak,
  TrainingPeaksCalendarSport,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { TriathlonFormatter } from '../runtime/formatter'
import { TRAINING_PEAK_SECONDS } from '../../../util/trainingpeaks-calendar'
import { trainingPeaksExecution } from '../../../util/trainingpeaks-execution'
import { triathlonActivityHref } from '../../../util/triathlon-date-route'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../../../util/triathlon-presentation'
import { createTriathlonFormatter } from '../runtime/formatter'
import { calendarDateLabel, calendarToday, calendarWeekdayLabel } from './display'
import {
  trainingAddDays,
  trainingAddMonths,
  trainingDayCovered,
  trainingDuration,
  trainingMonthDates,
  trainingMonthStart,
  trainingTotal,
  trainingWeekDates,
  trainingWeekLabel,
  trainingWeekStart,
  trainingWorkouts,
} from './training-display'
import { TrainingCalendarGate } from './TrainingCalendarGate'
import { TrainingDayNotes } from './TrainingDayNotes'
import { TrainingMonth } from './TrainingMonth'
import { TrainingStructureHover } from './TrainingStructureHover'
import { TrainingWorkoutCard, TrainingWorkoutDetail } from './TrainingWorkout'

const TRAININGPEAKS_URL = 'https://app.trainingpeaks.com/#calendar'

const PeriodPeaks = ({
  workouts,
  formatter,
  month,
}: {
  workouts: TrainingPeaksCalendarWorkout[]
  formatter: TriathlonFormatter
  month: boolean
}) => {
  const { text, number } = formatter
  const rows: {
    metric: 'heartRateBpm' | 'powerWatts'
    sport?: TrainingPeaksCalendarSport
    label: string
    unit: string
  }[] = [
    { metric: 'heartRateBpm', label: 'HR', unit: 'bpm' },
    { metric: 'powerWatts', sport: 'bike', label: 'bike', unit: 'W' },
    { metric: 'powerWatts', sport: 'run', label: 'run', unit: 'W' },
  ]
  const duration = (seconds: number): string =>
    seconds < 60 ? `${seconds} s` : `${seconds / 60} min`
  return (
    <div class="tri-training-week-peaks">
      <table
        class="tri-training-table tri-training-peaks-table"
        aria-label={text(month ? 'monthly peak averages' : 'weekly peak averages')}
        title={text(
          'HR requires a reported value for every second. Power uses each activity’s mean-max curve.',
        )}
      >
        <thead>
          <tr>
            <th scope="col">{text('peak averages')}</th>
            {TRAINING_PEAK_SECONDS.map(seconds => (
              <th key={seconds} scope="col">
                {duration(seconds)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.label}>
              <th scope="row">
                {text(row.label)} <span class="tri-training-peak-unit">{row.unit}</span>
              </th>
              {TRAINING_PEAK_SECONDS.map(seconds => {
                let best: {
                  activity: TrainingPeaksCalendarActivityLink
                  point: TrainingPeaksCalendarPeak
                  value: number
                } | null = null
                for (const workout of workouts) {
                  if (row.sport && workout.sport !== row.sport) continue
                  const activity = workout.activity
                  const point = activity?.peaks?.find(point => point.seconds === seconds)
                  const value = point?.[row.metric]
                  if (activity && point && value != null && (!best || value > best.value))
                    best = { activity, point, value }
                }
                const label = best
                  ? `${duration(seconds)} · ${text(row.label)}: ${number(best.value)} ${row.unit} · ${best.activity.title}${row.metric === 'heartRateBpm' ? ` · ${best.point.heartRateSource === 'garmin' ? 'Garmin' : 'Strava'}` : ''}`
                  : undefined
                const href = best
                  ? triathlonActivityHref(best.activity.date, best.activity.id)
                  : null
                return (
                  <td key={seconds}>
                    {best && href ? (
                      <a href={href} class="internal" aria-label={label}>
                        {number(best.value)}
                      </a>
                    ) : (
                      '—'
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const ExecutionLegend = ({ formatter, id }: { formatter: TriathlonFormatter; id: string }) => {
  const [active, setActive] = useState<string | null>(null)
  const { text, presentation } = formatter
  const percent = new Intl.NumberFormat(presentation.locale === 'fr' ? 'fr-CA' : 'en-CA', {
    style: 'percent',
    maximumFractionDigits: 0,
  })
  const items = [
    { status: 'on-target', label: 'on target', range: `80–${percent.format(1.2)}` },
    {
      status: 'off-target',
      label: 'off target',
      range: `50–<${percent.format(0.8)} · >120–${percent.format(1.5)}`,
    },
    {
      status: 'far-off-target',
      label: 'far off target',
      range: `<${percent.format(0.5)} · >${percent.format(1.5)}`,
    },
    { status: 'missed', label: 'missed', range: '×' },
    { status: 'unplanned', label: 'unplanned or unavailable', range: '—' },
  ]
  return (
    <ul class="tri-training-legend" aria-label={text('execution colors')}>
      {items.map(item => (
        <li
          key={item.status}
          data-execution={item.status}
          onPointerEnter={() => setActive(item.status)}
          onPointerLeave={() => setActive(null)}
        >
          <button
            type="button"
            aria-label={`${text(item.label)}: ${item.range}`}
            aria-describedby={`${id}-legend-${item.status}`}
            onFocus={() => setActive(item.status)}
            onBlur={() => setActive(null)}
            onClick={() => setActive(item.status)}
            onKeyDown={event => {
              if (event.key !== 'Escape' || active !== item.status) return
              event.preventDefault()
              event.stopPropagation()
              setActive(null)
            }}
          >
            <span class="tri-training-legend-swatch" aria-hidden="true" />
            <span aria-hidden="true">{item.range}</span>
          </button>
          <span
            class="tri-training-legend-tip"
            id={`${id}-legend-${item.status}`}
            role="tooltip"
            hidden={active !== item.status}
          >
            <strong>{text(item.label)}</strong>
            {text(
              'Execution colors compare actual with planned duration first, then distance, then TSS. Past uncompleted sessions are marked missed; future plans stay neutral.',
            )}{' '}
            {text('Colors use unrounded values.')}
          </span>
        </li>
      ))}
    </ul>
  )
}

export type TrainingView = 'list' | 'week' | 'month'

export interface TrainingCalendarViewProps {
  calendar: TrainingPeaksCalendar | null
  date: string
  id: string
  view?: TrainingView
  embedded?: boolean
  panel?: boolean
  presentation?: TriathlonPresentation
  loading?: boolean
  failed?: boolean
  selectedWorkoutId?: string | null
}

export const TrainingCalendarView = ({
  calendar,
  date,
  id,
  view = 'week',
  embedded = false,
  panel = false,
  presentation = DEFAULT_TRIATHLON_PRESENTATION,
  loading = false,
  failed = false,
  selectedWorkoutId = null,
}: TrainingCalendarViewProps) => {
  const formatter = createTriathlonFormatter(presentation)
  const { text, number } = formatter
  const today = calendarToday()
  const month = view === 'month'
  const start = month ? trainingMonthStart(date) : trainingWeekStart(date)
  const end = trainingAddDays(month ? trainingAddMonths(start, 1) : trainingAddDays(start, 7), -1)
  const dates = month ? trainingMonthDates(start) : trainingWeekDates(start)
  const visibleWorkouts = trainingWorkouts(calendar, dates[0] ?? start, dates.at(-1) ?? end)
  const workouts = visibleWorkouts.filter(workout => workout.date >= start && workout.date <= end)
  const completed = workouts.filter(workout => workout.status === 'completed')
  const selected = visibleWorkouts.find(workout => workout.id === selectedWorkoutId)
  const detailId = `${id}-detail`
  // Completed extras without any planned metric stay out of the planned count.
  const planned = workouts.filter(
    workout => trainingPeaksExecution(workout, today).status !== 'unplanned',
  )
  const periodDates = dates.filter(date => date >= start && date <= end)
  const covered = calendar
    ? periodDates.filter(date => trainingDayCovered(calendar, date)).length
    : 0
  const synced = calendar?.fetchedAt
  const sourceTime = synced ? new Date(synced) : null
  const syncedLabel =
    sourceTime && Number.isFinite(sourceTime.getTime())
      ? new Intl.DateTimeFormat(presentation.locale === 'fr' ? 'fr-CA' : 'en-CA', {
          month: 'short',
          day: 'numeric',
          hour: 'numeric',
          minute: '2-digit',
          timeZone: 'America/Toronto',
        }).format(sourceTime)
      : null
  const tss = (value: number | null): string => (value === null ? '—' : number(value, 0))
  const Heading = embedded || panel ? 'h2' : 'h1'
  const distance = (value: number | null): string =>
    value === null ? '—' : formatter.distance(value / 1000, 'bike')
  const message = loading
    ? 'Loading training calendar…'
    : failed
      ? 'Training calendar could not be loaded.'
      : !calendar
        ? 'TrainingPeaks calendar has not been imported yet.'
        : covered === 0
          ? month
            ? 'This month is outside the imported calendar.'
            : 'This week is outside the imported calendar.'
          : covered < periodDates.length
            ? month
              ? 'Part of this month has not been imported.'
              : 'Part of this week has not been imported.'
            : null
  return (
    <section
      class={`tri-calendar tri-training-calendar${embedded ? ' tri-calendar--embedded' : ''}`}
      aria-labelledby={`${id}-title`}
      data-calendar-detail={selected?.id}
      data-training-view={view}
    >
      <header class="tri-calendar-header tri-training-header">
        <div class="tri-calendar-title-group">
          <Heading id={`${id}-title`}>{text('training calendar')}</Heading>
          <p class="tri-calendar-summary">
            <a
              href={TRAININGPEAKS_URL}
              target="_blank"
              rel="noopener noreferrer"
              data-router-ignore
              data-no-popover
            >
              TrainingPeaks ↗
            </a>
            {syncedLabel && (
              <span class="tri-training-sync">
                {text('last synced')} <time dateTime={synced ?? undefined}>{syncedLabel}</time>
              </span>
            )}
          </p>
        </div>
        <div class="tri-calendar-actions tri-training-header-controls">
          <div class="tri-pred-controls tri-training-week-controls">
            <button type="button" class="g-text-button" data-training-today>
              {text('today')}
            </button>
            <div class="tri-training-jump" data-training-date-picker />
            <div class="tri-training-week-buttons">
              <button
                type="button"
                class="g-icon-button"
                data-training-shift="-1"
                aria-label={text(month ? 'previous month' : 'previous week')}
              >
                <svg class="g-icon" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M10 3.5 5.5 8l4.5 4.5"
                    stroke="currentColor"
                    stroke-width="1.6"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
              </button>
              <button
                type="button"
                class="g-icon-button"
                data-training-shift="1"
                aria-label={text(month ? 'next month' : 'next week')}
              >
                <svg class="g-icon" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M6 3.5 10.5 8 6 12.5"
                    stroke="currentColor"
                    stroke-width="1.6"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
              </button>
            </div>
          </div>
          {!embedded && (
            <div class="tri-calendar-view-controls" role="group" aria-label={text('calendar view')}>
              <button
                type="button"
                class="tri-calendar-icon"
                data-training-view-select="list"
                aria-pressed={view === 'list' ? 'true' : 'false'}
                aria-controls={`${id}-views`}
                aria-label={text('list')}
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M2 4h1m3 0h8M2 8h1m3 0h8M2 12h1m3 0h8" />
                </svg>
              </button>
              <button
                type="button"
                class="tri-calendar-icon"
                data-training-view-select="week"
                aria-pressed={view === 'week' ? 'true' : 'false'}
                aria-controls={`${id}-views`}
                aria-label={text('week')}
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M2.5 2.5h11v11h-11zM2.5 5.5h11M5.25 5.5v8M8 5.5v8M10.75 5.5v8" />
                </svg>
              </button>
              <button
                type="button"
                class="tri-calendar-icon"
                data-training-view-select="month"
                aria-pressed={month ? 'true' : 'false'}
                aria-controls={`${id}-views`}
                aria-label={text('month')}
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M2.5 2.5h11v11h-11zM2.5 5.5h11M2.5 9.5h11M6 5.5v8M10 5.5v8" />
                </svg>
              </button>
            </div>
          )}
          <button
            type="button"
            class="tri-calendar-icon tri-training-lock"
            data-training-lock
            aria-label={text('lock calendar')}
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <rect x="3.5" y="7" width="9" height="7" rx="1" />
              <path d="M5.5 7V4.5a2.5 2.5 0 0 1 5 0V7M8 10v1" />
            </svg>
          </button>
        </div>
      </header>
      <p class="tri-training-week-label" aria-live="polite" aria-atomic="true">
        {!month && <span>{text('week')} </span>}
        <time dateTime={start}>
          {month ? formatter.monthYear(start) : trainingWeekLabel(start, presentation.locale)}
        </time>
      </p>
      <div class="tri-training-week-overview">
        <table
          class="tri-training-table tri-training-week-table"
          title={text('Totals use reported TrainingPeaks values; missing values are left blank.')}
        >
          <thead>
            <tr>
              <td />
              <th scope="col">{text('planned')}</th>
              <th scope="col">{text('completed')}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">{text('sessions')}</th>
              <td>{covered > 0 ? number(planned.length) : '—'}</td>
              <td>{covered > 0 ? number(completed.length) : '—'}</td>
            </tr>
            <tr>
              <th scope="row">{text('duration')}</th>
              <td>{trainingDuration(trainingTotal(workouts, 'planned', 'durationSeconds'))}</td>
              <td>{trainingDuration(trainingTotal(completed, 'actual', 'durationSeconds'))}</td>
            </tr>
            <tr>
              <th scope="row">TSS</th>
              <td>{tss(trainingTotal(workouts, 'planned', 'tss'))}</td>
              <td>{tss(trainingTotal(completed, 'actual', 'tss'))}</td>
            </tr>
            <tr>
              <th scope="row">{text('distance')}</th>
              <td>{distance(trainingTotal(workouts, 'planned', 'distanceMeters'))}</td>
              <td>{distance(trainingTotal(completed, 'actual', 'distanceMeters'))}</td>
            </tr>
          </tbody>
        </table>
        <PeriodPeaks workouts={completed} formatter={formatter} month={month} />
      </div>
      {message && (
        <p class="tri-training-notice" role="status">
          {text(message)}
          {failed && (
            <button type="button" data-training-retry>
              {text('try again')}
            </button>
          )}
        </p>
      )}
      <TrainingStructureHover
        id={`${id}-structure-tip`}
        resetKey={`${start}:${view}:${selectedWorkoutId}:${presentation.locale}:${presentation.distance}`}
      >
        <div class="tri-calendar-body">
          <div class="tri-calendar-views" id={`${id}-views`}>
            {month ? (
              <TrainingMonth
                key={start}
                calendar={calendar}
                dates={dates}
                month={start}
                workouts={visibleWorkouts}
                formatter={formatter}
                today={today}
                id={`${id}-month`}
                detailId={detailId}
                selectedWorkoutId={selectedWorkoutId}
              />
            ) : (
              <div
                id={`${id}-week`}
                class="tri-training-week"
                aria-busy={loading ? 'true' : undefined}
              >
                {dates.map((date, index) => {
                  const sessions = workouts.filter(workout => workout.date === date)
                  const isCovered = calendar !== null && trainingDayCovered(calendar, date)
                  const isToday = date === today
                  return (
                    <section
                      key={date}
                      class="tri-training-day"
                      data-date={date}
                      data-today={isToday ? 'true' : undefined}
                      aria-labelledby={`${id}-day-${date}`}
                    >
                      <header class="tri-training-day-header">
                        <h3 id={`${id}-day-${date}`}>
                          <time
                            dateTime={date}
                            aria-label={`${calendarWeekdayLabel(index, presentation.locale)} ${calendarDateLabel(date, null, 'full', presentation.locale)}`}
                          >
                            {calendarDateLabel(date, null, 'short', presentation.locale)}
                          </time>
                        </h3>
                        <TrainingDayNotes
                          id={`${id}-notes-${date}`}
                          notes={calendar?.notes?.filter(note => note.date === date) ?? []}
                          formatter={formatter}
                        />
                        {isToday && <span class="tri-training-today-label">{text('today')}</span>}
                      </header>
                      <div class="tri-training-day-sessions">
                        {sessions.length > 0 ? (
                          sessions.map(workout => (
                            <TrainingWorkoutCard
                              key={workout.id}
                              workout={workout}
                              formatter={formatter}
                              today={today}
                              zones={calendar}
                              detailId={detailId}
                              selected={selected?.id === workout.id}
                            />
                          ))
                        ) : (
                          <p class="tri-training-empty" data-covered={isCovered ? 'true' : 'false'}>
                            {text(isCovered ? 'No sessions.' : 'Not synced.')}
                          </p>
                        )}
                      </div>
                    </section>
                  )
                })}
              </div>
            )}
          </div>
          <aside
            id={detailId}
            class="tri-calendar-detail tri-training-detail"
            role="region"
            aria-label={selected?.title}
            aria-hidden={!selected}
          >
            {selected && (
              <TrainingWorkoutDetail
                workout={selected}
                formatter={formatter}
                today={today}
                zones={calendar}
                sessions={visibleWorkouts.filter(workout => workout.date === selected.date)}
              />
            )}
          </aside>
        </div>
      </TrainingStructureHover>
      <ExecutionLegend formatter={formatter} id={id} />
    </section>
  )
}

export const TrainingCalendar = ({
  id,
  embedded,
  panel,
}: {
  id: string
  embedded: boolean
  panel: boolean
}) => {
  return (
    <div
      data-training-calendar
      data-training-id={id}
      data-training-embedded={String(embedded)}
      data-training-panel={String(panel)}
      data-training-date={calendarToday()}
    >
      <div data-training-content>
        <TrainingCalendarGate id={id} />
      </div>
    </div>
  )
}
