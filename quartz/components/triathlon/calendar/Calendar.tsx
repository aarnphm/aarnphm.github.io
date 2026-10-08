import type { ComponentChildren } from 'preact'
import type {
  TriathlonCalendar,
  TriathlonCalendarEvent,
  TriathlonCalendarSeries,
} from '../../../util/triathlon-calendar'
import type { TriathlonRenderData } from '../render-data'
import type { CalendarDatePart } from './display'
import {
  calendarRaceStart,
  calendarScheduleDates,
  calendarScheduleEdition,
  TRIATHLON_CALENDAR_LEGS,
} from '../../../util/triathlon-calendar'
import { CalendarResults } from './CalendarResults'
import {
  courseFigure,
  formatDistance,
  PROFILE_HEIGHT,
  PROFILE_WIDTH,
  profileDomain,
  profilePaths,
  ROUTE_SIZE,
} from './course'
import {
  calendarDateLabel,
  calendarMonthLabel,
  calendarToday,
  calendarWeekdayLabel,
} from './display'
import { RaceProjection } from './RaceProjection'
import { TrainingCalendar } from './TrainingCalendar'

const MONTHS = Array.from({ length: 12 }, (_, month) => month)
const WEEKDAYS = Array.from({ length: 7 }, (_, day) => day)
const SERIES_LABEL: Record<TriathlonCalendarSeries, string> = {
  ironman: 'IRONMAN',
  t100: 'T100',
  supertri: 'Supertri',
  hyrox: 'HYROX',
  tcs: 'TCS',
  b4bh: 'B4BH',
  other: 'other',
}
const pad = (value: number): string => String(value).padStart(2, '0')
const eventMonth = (event: TriathlonCalendarEvent): number =>
  event.date ? Number(event.date.slice(5, 7)) - 1 : -1
const celsius = ([low, high]: [number, number]): string =>
  low === high ? `${low} °C` : `${low}–${high} °C`

const LocalDate = ({
  date,
  end = null,
  part,
}: {
  date: string
  end?: string | null
  part: CalendarDatePart
}) => (
  <time
    dateTime={date}
    data-calendar-date={date}
    data-calendar-end={end ?? undefined}
    data-calendar-part={part}
  >
    {calendarDateLabel(date, end, part)}
  </time>
)

const CalendarDate = ({ event }: { event: TriathlonCalendarEvent }) => {
  if (!event.date)
    return (
      <span class="tri-calendar-date tri-calendar-pending" data-i18n="date pending">
        date pending
      </span>
    )
  return (
    <time
      class="tri-calendar-date"
      dateTime={event.date}
      aria-label={calendarDateLabel(event.date, event.endDate, 'full')}
      data-calendar-date-label={event.date}
      data-calendar-end={event.endDate ?? undefined}
    >
      <span
        class="tri-calendar-date-month"
        data-calendar-date={event.date}
        data-calendar-end={event.endDate ?? undefined}
        data-calendar-part="month"
      >
        {calendarDateLabel(event.date, event.endDate, 'month')}
      </span>
      <span class="tri-calendar-date-day">
        {calendarDateLabel(event.date, event.endDate, 'day')}
      </span>
      <span
        class="tri-calendar-date-weekday"
        data-calendar-date={event.date}
        data-calendar-end={event.endDate ?? undefined}
        data-calendar-part="weekday"
      >
        {calendarDateLabel(event.date, event.endDate, 'weekday')}
      </span>
    </time>
  )
}

const EventRow = ({
  event,
  next,
  prefix,
  detailId,
}: {
  event: TriathlonCalendarEvent
  next: boolean
  prefix: string
  detailId: string
}) => {
  const schedule = event.details?.schedule
  const days = calendarScheduleDates(event)
  const start = calendarRaceStart(event)
  return (
    <li
      class="tri-calendar-event"
      id={`${prefix}race-${event.id}`}
      data-calendar-event={`${prefix}race-${event.id}`}
      data-calendar-start={event.date ?? undefined}
      data-calendar-next={next ? 'true' : undefined}
      data-kind={event.kind}
      data-series={event.series}
      data-calendar-card={event.id}
    >
      <CalendarDate event={event} />
      <div class="tri-calendar-event-body">
        <div class="tri-calendar-event-heading">
          <button
            type="button"
            data-calendar-card={event.id}
            data-calendar-card-open
            aria-expanded="false"
            aria-controls={detailId}
          >
            {event.name}
          </button>
          <span class="tri-calendar-next" data-i18n="next up" hidden={!next}>
            next up
          </span>
        </div>
        <p class="tri-calendar-event-meta">
          <span>{event.location}</span>
          {event.note && <span>{event.note}</span>}
        </p>
        {schedule && days.length > 0 && (
          <div
            class="tri-calendar-schedule-days"
            role="group"
            aria-label="schedule"
            data-i18n-aria-label="schedule"
          >
            {days.map(day => (
              <button
                type="button"
                class="tri-calendar-schedule-day"
                data-calendar-card-open
                aria-expanded="false"
                aria-controls={detailId}
                data-race={day.race ? 'true' : undefined}
                data-calendar-card={event.id}
                data-calendar-card-date={day.date}
              >
                <LocalDate date={day.date} part="short" />
                {day.race && start && <span class="tri-calendar-start">{start.start}</span>}
              </button>
            ))}
          </div>
        )}
      </div>
      <div class="tri-calendar-event-labels">
        <span class="tri-calendar-format">{event.format}</span>
        {event.participated && (
          <span class="tri-calendar-participated" data-i18n="participated">
            participated
          </span>
        )}
      </div>
    </li>
  )
}

interface DayMark {
  event: TriathlonCalendarEvent
  race: boolean
}

/**
 * Every event touching a date, race days first so the cell takes the racing series' colour. The
 * cell's card lists each event's schedule for that date.
 */
const dayMarks = (events: readonly TriathlonCalendarEvent[]): Map<string, DayMark[]> => {
  const marks = new Map<string, DayMark[]>()
  const mark = (date: string, entry: DayMark): void => {
    const entries = marks.get(date) ?? []
    if (entries.some(existing => existing.event.id === entry.event.id)) return
    entries.push(entry)
    marks.set(date, entries)
  }
  for (const event of events) {
    if (!event.date) continue
    const cursor = new Date(`${event.date}T00:00:00.000Z`)
    const end = event.endDate ?? event.date
    for (let date = event.date; date <= end; date = cursor.toISOString().slice(0, 10)) {
      mark(date, { event, race: true })
      cursor.setUTCDate(cursor.getUTCDate() + 1)
    }
  }
  for (const event of events)
    for (const day of calendarScheduleDates(event)) mark(day.date, { event, race: false })
  return marks
}

const YearMonth = ({
  month,
  year,
  id,
  prefix,
  linkPrefix,
  marks,
}: {
  month: number
  year: number
  id: string
  prefix: string
  linkPrefix: string
  marks: Map<string, DayMark[]>
}) => {
  const firstWeekday = (new Date(Date.UTC(year, month, 1)).getUTCDay() + 6) % 7
  const days = new Date(Date.UTC(year, month + 1, 0)).getUTCDate()
  return (
    <section class="tri-calendar-month" aria-labelledby={`${id}-month-${month}`}>
      <h2 id={`${id}-month-${month}`}>
        <span data-calendar-month={month}>{calendarMonthLabel(year, month)}</span>
        <span class="tri-calendar-month-number" aria-hidden="true">
          {pad(month + 1)}
        </span>
      </h2>
      <div class="tri-calendar-days">
        {WEEKDAYS.map(day => (
          <abbr
            class="tri-calendar-weekday"
            title={calendarWeekdayLabel(day)}
            data-calendar-weekday={day}
          >
            {calendarWeekdayLabel(day).slice(0, 1)}
          </abbr>
        ))}
        {/* Six week rows in every month keep all twelve months the same height. */}
        {Array.from({ length: 42 }, (_, cell) => {
          const day = cell - firstWeekday + 1
          if (day < 1 || day > days) return <span class="tri-calendar-day" aria-hidden="true" />
          const date = `${year}-${pad(month + 1)}-${pad(day)}`
          const entries = marks.get(date)
          if (!entries) return <span class="tri-calendar-day">{day}</span>
          const [mark] = entries
          const names = entries.map(entry => entry.event.name).join(', ')
          return (
            <a
              class="tri-calendar-day"
              href={`#${linkPrefix}race-${mark.event.id}`}
              data-calendar-target={`${prefix}race-${mark.event.id}`}
              data-calendar-day={date}
              data-calendar-event-name={names}
              data-series={mark.event.series}
              data-race={mark.race ? 'true' : undefined}
              data-calendar-card={entries.map(entry => entry.event.id).join(' ')}
              data-calendar-card-date={date}
              aria-label={`${calendarDateLabel(date, null, 'full')}: ${names}`}
            >
              {day}
            </a>
          )
        })}
      </div>
    </section>
  )
}

const sourceLabel = (url: string): string => {
  const { hostname, pathname } = new URL(url)
  return `${hostname.replace(/^www\./, '')}${pathname.replace(/\/$/, '')}`
}

const ExternalLink = ({ href, children }: { href: string; children: ComponentChildren }) => (
  <a href={href} target="_blank" rel="noopener noreferrer" data-no-popover>
    {children}
    <span class="tri-calendar-external" aria-hidden="true">
      ↗
    </span>
  </a>
)

/**
 * One template per event serves both surfaces: the hover card drops `data-card-detail` parts and
 * keeps one day, and the detail beside the views drops the `data-card-preview` head and keeps all.
 */
const EventCard = ({
  event,
  year,
  fetched,
}: {
  event: TriathlonCalendarEvent
  year: number
  fetched: string | null
}) => {
  const details = event.details
  const courses = details?.course ?? []
  const figure = courseFigure(courses)
  const domain = profileDomain(courses)
  const legs = TRIATHLON_CALENDAR_LEGS.filter(
    leg => details?.terrain[leg] || courses.some(course => course.leg === leg),
  )
  const days = calendarScheduleDates(event)
  const edition = details?.schedule ? calendarScheduleEdition(details.schedule) : year
  const start = calendarRaceStart(event)
  // A total needs every leg; a partial sum would read as the race distance.
  const total = TRIATHLON_CALENDAR_LEGS.every(leg => courses.some(course => course.leg === leg))
    ? courses.reduce((sum, course) => sum + (course.raceDistanceM ?? course.distanceM), 0)
    : null
  const gain = courses.reduce((sum, course) => sum + (course.elevationGainM ?? 0), 0)
  const courseEditions = [
    ...new Set(
      courses.flatMap(course =>
        course.edition && course.edition !== year ? [course.edition] : [],
      ),
    ),
  ]
  const detailsFetched = details?.fetched ?? fetched
  const linked = new Set([
    event.url,
    details?.schedule?.source,
    details?.venueMap,
    details?.athleteGuide,
    ...courses.flatMap(c => [c.source, c.mapUrl]),
  ])
  const pages = (details?.sources ?? []).filter(source => !linked.has(source))
  const kept = Object.entries(details?.kept ?? {})
  const where = details?.venue ?? event.location
  return (
    <template
      data-calendar-card-template={event.id}
      data-card-start={event.date ?? undefined}
      data-card-end={event.endDate ?? event.date ?? undefined}
    >
      <article class="tri-calendar-card" data-series={event.series}>
        <div class="tri-calendar-card-head" data-card-preview>
          <span class="tri-calendar-card-series">
            {SERIES_LABEL[event.series]} · {event.format}
          </span>
          <strong>{event.name}</strong>
          <span class="tri-calendar-card-where">
            {event.date ? (
              <LocalDate date={event.date} end={event.endDate} part="full" />
            ) : (
              <span data-i18n="date pending">date pending</span>
            )}
            {' · '}
            {where}
          </span>
          {event.participated && <span data-i18n="participated">participated</span>}
        </div>
        <p class="tri-calendar-card-meta" data-card-detail>
          <span class="tri-calendar-card-series">
            {SERIES_LABEL[event.series]} · {event.format}
          </span>
          <span>{where}</span>
          {event.participated && <span data-i18n="participated">participated</span>}
          {event.note && <span>{event.note}</span>}
        </p>
        {event.results && <CalendarResults results={event.results} />}
        <dl class="tri-calendar-card-stats" data-card-detail>
          {event.date && (
            <div data-card-countdown={event.date} hidden>
              <dt data-i18n="days to go">days to go</dt>
              <dd />
            </div>
          )}
          {start && (
            <div>
              <dt data-i18n="race start">race start</dt>
              <dd>{start.start}</dd>
            </div>
          )}
          {total !== null && (
            <div>
              <dt data-i18n="course">course</dt>
              <dd>{formatDistance(total)}</dd>
            </div>
          )}
          {gain > 0 && (
            <div>
              <dt data-i18n="elevation gain">elevation gain</dt>
              <dd>+{gain} m</dd>
            </div>
          )}
        </dl>
        {courseEditions.length > 0 && (
          <p class="tri-calendar-card-conditions" data-card-course>
            <span data-i18n="course reference">course reference</span>
            <span>{courseEditions.join(', ')}</span>
          </p>
        )}
        {legs.length > 0 && (
          <div class="tri-calendar-card-course" data-card-course>
            {figure && (
              <svg
                class="tri-calendar-route"
                viewBox={`0 0 ${ROUTE_SIZE} ${ROUTE_SIZE}`}
                aria-hidden="true"
              >
                {figure.legs.map(leg => (
                  <path d={leg.d} data-leg={leg.leg} />
                ))}
                <circle
                  class="tri-calendar-route-start"
                  cx={figure.start[0]}
                  cy={figure.start[1]}
                  r="2.4"
                />
                <rect
                  class="tri-calendar-route-finish"
                  x={figure.finish[0] - 2.2}
                  y={figure.finish[1] - 2.2}
                  width="4.4"
                  height="4.4"
                />
              </svg>
            )}
            <dl class="tri-calendar-legs">
              {legs.map(leg => {
                const course = courses.find(entry => entry.leg === leg)
                const terrain = details?.terrain[leg]
                const profile =
                  course && leg !== 'swim' ? profilePaths(course.profile, domain) : null
                return (
                  <div class="tri-calendar-leg" data-leg={leg}>
                    <dt data-i18n={leg}>{leg}</dt>
                    <dd>{course && formatDistance(course.raceDistanceM ?? course.distanceM)}</dd>
                    <dd class="tri-calendar-leg-profile">
                      {profile && course && (
                        <svg
                          viewBox={`0 0 ${PROFILE_WIDTH} ${PROFILE_HEIGHT}`}
                          preserveAspectRatio="none"
                          aria-hidden="true"
                        >
                          <path class="tri-calendar-profile-area" d={profile.area} />
                          <path class="tri-calendar-profile-line" d={profile.line} />
                          {course.aidM.map(distance => {
                            const x = Math.round((distance / course.distanceM) * PROFILE_WIDTH)
                            return (
                              <path
                                class="tri-calendar-profile-aid"
                                d={`M${x} ${PROFILE_HEIGHT}V${PROFILE_HEIGHT - 4}`}
                              />
                            )
                          })}
                        </svg>
                      )}
                    </dd>
                    <dd class="tri-calendar-leg-note">
                      {course?.elevationGainM ? (
                        `+${course.elevationGainM} m`
                      ) : terrain ? (
                        <span data-i18n={`${terrain} terrain`}>{terrain}</span>
                      ) : null}
                    </dd>
                    {course && (
                      <dd class="tri-calendar-leg-source" data-card-detail>
                        <ExternalLink href={course.source}>{course.title}</ExternalLink>
                        {course.raceDistanceM !== null &&
                          course.raceDistanceM !== course.distanceM && (
                            <span>
                              <span data-i18n="mapped distance">mapped distance</span>{' '}
                              {formatDistance(course.distanceM)}
                            </span>
                          )}
                        {course.laps !== null && (
                          <span>
                            {course.laps}{' '}
                            <span data-i18n={course.laps === 1 ? 'lap' : 'laps'}>
                              {course.laps === 1 ? 'lap' : 'laps'}
                            </span>
                          </span>
                        )}
                        {course.aidStationsPerLap !== null && (
                          <span>
                            {course.aidStationsPerLap}{' '}
                            <span
                              data-i18n={
                                course.aidStationsPerLap === 1
                                  ? 'aid station per lap'
                                  : 'aid stations per lap'
                              }
                            >
                              {course.aidStationsPerLap === 1
                                ? 'aid station per lap'
                                : 'aid stations per lap'}
                            </span>
                          </span>
                        )}
                        {course.aidM.length > 0 && (
                          <span>
                            {course.aidM.length} <span data-i18n="aid stations">aid stations</span>
                          </span>
                        )}
                        {course.elevationGainM && terrain ? (
                          <span data-i18n={`${terrain} terrain`}>{terrain}</span>
                        ) : null}
                        {course.mapUrl && (
                          <ExternalLink href={course.mapUrl}>
                            <span data-i18n="course map">course map</span>
                          </ExternalLink>
                        )}
                        {course.gpxUrl && (
                          <a href={course.gpxUrl} download data-no-popover data-router-ignore>
                            <span data-i18n="download GPX">download GPX</span>
                          </a>
                        )}
                      </dd>
                    )}
                  </div>
                )
              })}
            </dl>
          </div>
        )}
        {(details?.airC || details?.waterC) && (
          <p class="tri-calendar-card-conditions" data-card-course>
            {details.airC && (
              <span>
                <span data-i18n="air">air</span> {celsius(details.airC)}
              </span>
            )}
            {details.waterC && (
              <span>
                <span data-i18n="water">water</span> {celsius(details.waterC)}
              </span>
            )}
          </p>
        )}
        {days.map(day => (
          <section
            class="tri-calendar-card-day"
            data-card-date={day.date}
            data-card-race={day.race ? 'true' : undefined}
          >
            <h3>
              <LocalDate date={day.date} part="short" />
              {day.race && (
                <span class="tri-calendar-card-race" data-i18n="race day">
                  race day
                </span>
              )}
              {edition !== year && (
                <span class="tri-calendar-card-edition" data-calendar-edition={edition}>
                  {edition} schedule
                </span>
              )}
            </h3>
            <ol>
              {day.items.map((item, index) => (
                <li data-race={item.race ? 'true' : undefined}>
                  <span class="tri-calendar-card-time">
                    {item.end && !item.race ? `${item.start}–${item.end}` : item.start}
                  </span>
                  <span>
                    {item.activity}
                    {item.location && item.location !== day.items[index - 1]?.location && (
                      <span class="tri-calendar-card-place">{item.location}</span>
                    )}
                  </span>
                </li>
              ))}
            </ol>
          </section>
        ))}
        {details?.schedulePending && (
          <p class="tri-calendar-card-pending" data-i18n="schedule not published">
            schedule not published
          </p>
        )}
        <div class="tri-calendar-card-sources" data-card-detail>
          <div class="tri-calendar-card-links">
            <ExternalLink href={event.url}>
              <span data-i18n="event website">event website</span>
            </ExternalLink>
            {details?.schedule && (
              <ExternalLink href={details.schedule.source}>
                <span data-i18n="schedule">schedule</span>
              </ExternalLink>
            )}
            {details?.venueMap && (
              <ExternalLink href={details.venueMap}>
                <span data-i18n="venue map">venue map</span>
              </ExternalLink>
            )}
            {details?.athleteGuide && (
              <ExternalLink href={details.athleteGuide}>
                <span data-i18n="athlete guide">athlete guide</span>
              </ExternalLink>
            )}
          </div>
          {pages.length > 0 && (
            <p>
              <span data-i18n="sources">sources</span>
              {pages.map(page => (
                <ExternalLink href={page}>{sourceLabel(page)}</ExternalLink>
              ))}
            </p>
          )}
          {details && detailsFetched && (
            <p>
              <span data-i18n="details fetched">details fetched</span>
              <LocalDate date={detailsFetched} part="full" />
            </p>
          )}
          {kept.map(([field, date]) => (
            <p>
              <span data-i18n="kept from an earlier sync">kept from an earlier sync</span>
              <span>
                {field} · <LocalDate date={date} part="full" />
              </span>
            </p>
          ))}
        </div>
      </article>
    </template>
  )
}

// The overlay renders these in its panel bar beside the close button; pages and embeds keep them
// above the calendar.
export const CalendarSourceControls = ({ id, icons = false }: { id: string; icons?: boolean }) => (
  <div
    class={`tri-calendar-source-controls${icons ? ' tri-calendar-source-controls--icons' : ''}`}
    role="group"
    aria-label="calendar source"
    data-i18n-aria-label="calendar source"
  >
    <button
      type="button"
      class={icons ? 'tri-calendar-icon' : 'tri-calendar-source-tab'}
      data-calendar-source-select="races"
      aria-pressed="true"
      aria-controls={`${id}-races`}
      aria-label={icons ? 'race calendar' : 'race'}
      data-i18n-aria-label={icons ? 'race calendar' : 'race'}
    >
      {icons ? (
        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <path d="M3.5 15V1.5M3.5 2h10v7.5h-10" />
          <path
            class="tri-calendar-icon-fill"
            d="M3.5 2H6v2.5H3.5zM8.5 2H11v2.5H8.5zM6 4.5h2.5V7H6zM11 4.5h2.5V7H11zM3.5 7H6v2.5H3.5zM8.5 7H11v2.5H8.5z"
          />
        </svg>
      ) : (
        <span data-i18n="race">race</span>
      )}
    </button>
    <button
      type="button"
      class={icons ? 'tri-calendar-icon' : 'tri-calendar-source-tab'}
      data-calendar-source-select="training"
      aria-pressed="false"
      aria-controls={`${id}-training`}
      aria-label={icons ? 'training calendar' : 'training'}
      data-i18n-aria-label={icons ? 'training calendar' : 'training'}
    >
      {icons ? (
        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <path d="M6.25 1.5h3.5M8 1.5V4M12 5.25l1-1" />
          <circle cx="8" cy="9.25" r="5.25" />
          <path class="tri-calendar-icon-fill" d="M8 9.25V5.75A3.5 3.5 0 0 1 11.03 11z" />
        </svg>
      ) : (
        <span data-i18n="training">training</span>
      )}
    </button>
  </div>
)

interface CalendarPanelProps {
  renderData?: TriathlonRenderData
  calendar?: TriathlonCalendar | null
  calendars?: readonly TriathlonCalendar[]
  year?: number
  id?: string
  embedded?: boolean
  panel?: boolean
  view?: 'list' | 'year'
}

export const CalendarPanel = ({
  renderData,
  calendar = renderData?.calendar,
  calendars = renderData?.calendars ?? (calendar ? [calendar] : []),
  year = calendar?.year,
  id = 'tri-calendar',
  embedded = false,
  panel = false,
  view = 'list',
}: CalendarPanelProps) => {
  const selectedYear =
    calendars.find(calendar => calendar.year === year)?.year ?? calendars.at(-1)?.year
  return (
    <div
      id={id}
      class="tri-calendar-set"
      data-calendar-set
      data-calendar-default-year={selectedYear}
      data-calendar-source="races"
      data-calendar-embedded={embedded ? 'true' : undefined}
    >
      {!panel &&
        (embedded ? (
          <CalendarSourceControls id={id} icons />
        ) : (
          <div class="tri-ana-bar">
            <CalendarSourceControls id={id} />
          </div>
        ))}
      <div id={`${id}-races`} data-calendar-source-panel="races">
        {calendars.length > 0 ? (
          calendars.map(calendar => (
            <CalendarSeason
              calendar={calendar}
              calendars={calendars}
              id={`${id}-${calendar.year}`}
              embedded={embedded}
              panel={panel}
              view={view}
              hidden={calendar.year !== selectedYear}
            />
          ))
        ) : (
          <section class="tri-calendar">
            <p data-i18n="No events planned.">No events planned.</p>
          </section>
        )}
      </div>
      <div id={`${id}-training`} data-calendar-source-panel="training" hidden inert>
        <TrainingCalendar id={`${id}-training-calendar`} embedded={embedded} panel={panel} />
      </div>
    </div>
  )
}

const CalendarSeason = ({
  calendar,
  calendars,
  id,
  embedded,
  panel,
  view,
  hidden,
}: {
  calendar: TriathlonCalendar
  calendars: readonly TriathlonCalendar[]
  id: string
  embedded: boolean
  panel: boolean
  view: 'list' | 'year'
  hidden: boolean
}) => {
  const next = calendar.events.find(event => event.date && event.date >= calendarToday())
  const series = new Map<TriathlonCalendarSeries, number>()
  for (const event of calendar.events) series.set(event.series, (series.get(event.series) ?? 0) + 1)
  const pending = calendar.events.filter(event => !event.date)
  const marks = dayMarks(calendar.events)
  const prefix = embedded ? `${id}-` : ''
  const linkPrefix = panel ? 'calendar-' : prefix
  const Heading = embedded || panel ? 'h2' : 'h1'
  return (
    <section
      id={id}
      class={`tri-calendar${embedded ? ' tri-calendar--embedded' : ''}`}
      data-calendar-year={calendar.year}
      data-calendar-view={view}
      data-calendar-embedded={embedded ? 'true' : undefined}
      aria-labelledby={`${id}-title`}
      hidden={hidden}
      inert={hidden}
    >
      <header class="tri-calendar-header">
        <div class="tri-calendar-title-group">
          <Heading id={`${id}-title`} data-i18n="race calendar">
            race calendar
          </Heading>
          <p class="tri-calendar-summary">
            <strong>{calendar.events.length}</strong>{' '}
            <span data-i18n={calendar.events.length === 1 ? 'event' : 'events'}>
              {calendar.events.length === 1 ? 'event' : 'events'}
            </span>
          </p>
        </div>
        <div class="tri-calendar-actions">
          <div class="tri-calendar-season-picker">
            <span id={`${id}-year-label`} data-calendar-year-label data-i18n="calendar year" hidden>
              calendar year
            </span>
            <button
              class="tri-calendar-season-trigger"
              type="button"
              data-calendar-year-select
              aria-labelledby={`${id}-year-label ${id}-year-value`}
              aria-haspopup="listbox"
              aria-expanded="false"
              aria-controls={`${id}-years`}
            >
              <span id={`${id}-year-value`} class="tri-calendar-season-value">
                {calendar.year}
              </span>
              <svg
                class="tri-calendar-season-chevron"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
                focusable="false"
              >
                <path
                  d="m4 6 4 4 4-4"
                  stroke="currentColor"
                  stroke-width="1.4"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </button>
            <div
              id={`${id}-years`}
              class="tri-calendar-season-menu"
              role="listbox"
              aria-labelledby={`${id}-year-label`}
              hidden
            >
              {calendars.map(season => (
                <button
                  class="tri-calendar-season-option"
                  type="button"
                  role="option"
                  data-calendar-year-option={season.year}
                  aria-selected={season.year === calendar.year}
                  tabIndex={season.year === calendar.year ? 0 : -1}
                >
                  <span class="tri-calendar-season-check" aria-hidden="true">
                    ✓
                  </span>
                  <span class="tri-calendar-season-option-value">{season.year}</span>
                </button>
              ))}
            </div>
          </div>
          {!embedded && (
            <div
              class="tri-calendar-view-controls"
              role="group"
              aria-label="calendar view"
              data-i18n-aria-label="calendar view"
            >
              <button
                type="button"
                class="tri-calendar-icon"
                data-calendar-select="list"
                aria-pressed={view === 'list' ? 'true' : 'false'}
                aria-controls={`${id}-list`}
                aria-label="list"
                data-i18n-aria-label="list"
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M2 4h1m3 0h8M2 8h1m3 0h8M2 12h1m3 0h8" />
                </svg>
              </button>
              <button
                type="button"
                class="tri-calendar-icon"
                data-calendar-select="year"
                aria-pressed={view === 'year' ? 'true' : 'false'}
                aria-controls={`${id}-year`}
                aria-label="year"
                data-i18n-aria-label="year"
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M2.5 2.5h11v11h-11zM2.5 6h11M6 6v7.5M10 6v7.5M2.5 10h11" />
                </svg>
              </button>
            </div>
          )}
          <a
            class="tri-calendar-icon tri-calendar-download"
            href={`/triathlon/calendar/${calendar.year}.ics`}
            download={`races-${calendar.year}.ics`}
            data-no-popover
            data-router-ignore
            aria-label="download calendar"
            data-i18n-aria-label="download calendar"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M8 2v8m-3-3 3 3 3-3M3 10v3h10v-3" />
            </svg>
          </a>
        </div>
      </header>
      <nav class="tri-calendar-strip" aria-label={String(calendar.year)}>
        {MONTHS.map(month => {
          const events = calendar.events.filter(event => eventMonth(event) === month)
          const contents = (
            <>
              <span data-calendar-month={month} data-calendar-short>
                {calendarMonthLabel(calendar.year, month, 'en', true)}
              </span>
              <span class="tri-calendar-strip-dots" aria-hidden="true">
                {events.map(event => (
                  <span class="tri-calendar-dot" data-series={event.series} />
                ))}
              </span>
            </>
          )
          return events[0] ? (
            <a
              href={`#${linkPrefix}race-${events[0].id}`}
              data-calendar-target={`${prefix}race-${events[0].id}`}
              data-calendar-card={events.map(event => event.id).join(' ')}
            >
              {contents}
            </a>
          ) : (
            <span class="tri-calendar-strip-empty">{contents}</span>
          )
        })}
      </nav>
      <div class="tri-calendar-legend">
        {Array.from(series, ([name, count]) => (
          <span data-series={name}>
            <span class="tri-calendar-dot" aria-hidden="true" />
            {count} {SERIES_LABEL[name]}
          </span>
        ))}
      </div>
      <div class="tri-calendar-body">
        <div class="tri-calendar-views">
          <div
            id={`${id}-list`}
            class="tri-calendar-view"
            data-calendar-panel="list"
            inert={view !== 'list'}
          >
            {calendar.events.length === 0 && (
              <p data-i18n="No events planned.">No events planned.</p>
            )}
            <ol class="tri-calendar-list">
              {calendar.events.map(event => (
                <EventRow
                  event={event}
                  next={next?.id === event.id}
                  prefix={prefix}
                  detailId={`${id}-detail`}
                />
              ))}
            </ol>
          </div>
          <div
            id={`${id}-year`}
            class="tri-calendar-view"
            data-calendar-panel="year"
            inert={view !== 'year'}
          >
            <div class="tri-calendar-year-grid">
              {MONTHS.map(month => (
                <YearMonth
                  month={month}
                  year={calendar.year}
                  id={id}
                  prefix={prefix}
                  linkPrefix={linkPrefix}
                  marks={marks}
                />
              ))}
            </div>
            {pending.length > 0 && (
              <div class="tri-calendar-undated">
                <h2 data-i18n="date pending">date pending</h2>
                {pending.map(event => (
                  <a
                    href={`#${linkPrefix}race-${event.id}`}
                    data-calendar-target={`${prefix}race-${event.id}`}
                  >
                    {event.name}
                  </a>
                ))}
              </div>
            )}
          </div>
          <div class="tri-calendar-projection" data-race-projection-slot hidden />
        </div>
        <div class="tri-calendar-preview-empty">
          <h2 data-i18n="race preview">race preview</h2>
          <p data-i18n="Select a race to see its course and schedule.">
            Select a race to see its course and schedule.
          </p>
          {next && (
            <a
              href={`#${linkPrefix}race-${next.id}`}
              data-calendar-target={`${prefix}race-${next.id}`}
              data-calendar-card={next.id}
            >
              <span data-i18n="next up">next up</span>
              <strong>{next.name}</strong>
            </a>
          )}
        </div>
        <div id={`${id}-detail`} class="tri-calendar-detail" role="region" aria-hidden="true" />
      </div>
      {calendar.events.map(event => (
        <>
          <EventCard event={event} year={calendar.year} fetched={calendar.fetched} />
          <template data-race-projection-template={event.id}>
            <RaceProjection event={event} />
          </template>
        </>
      ))}
      <div id={`${id}-card`} class="tri-calendar-pop" popover="manual" role="tooltip" />
      {calendar.checked && (
        <footer class="tri-calendar-footer">
          <span data-i18n="dates checked">dates checked</span>{' '}
          <LocalDate date={calendar.checked} part="full" />
        </footer>
      )}
    </section>
  )
}
