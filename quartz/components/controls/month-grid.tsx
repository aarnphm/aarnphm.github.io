import type { ComponentChildren, KeyboardEventHandler, MouseEventHandler } from 'preact'

export interface MonthGridClasses {
  root: string
  weekdays: string
  grid: string
  day: string
  dayHeader: string
}

const DEFAULT_CLASSES: MonthGridClasses = {
  root: 'g-calendar',
  weekdays: 'g-calendar-weekdays',
  grid: 'g-calendar-grid',
  day: 'g-calendar-day',
  dayHeader: 'g-calendar-day-header',
}

/**
 * A month of days in weekday columns. Each day is a `section` named by its date heading; the
 * host renders what the day holds. `dates` is a whole number of weeks.
 */
export const MonthGrid = ({
  id,
  dates,
  month,
  today,
  classes = DEFAULT_CLASSES,
  rowMinHeight = '5.5rem',
  weekdayLabel,
  dayLabel,
  dayAttributes,
  dayHeader,
  renderDay,
  onKeyDown,
  onClickCapture,
  children,
}: {
  id: string
  dates: string[]
  /** Any `YYYY-MM…` value in the shown month; days from other months get `data-outside`. */
  month: string
  today: string
  classes?: MonthGridClasses
  rowMinHeight?: string
  weekdayLabel: (date: string) => string
  /** Accessible name of a day, such as "wednesday october 7". */
  dayLabel: (date: string, index: number) => string
  dayAttributes?: (date: string) => Record<string, string | undefined>
  dayHeader?: (date: string) => ComponentChildren
  renderDay: (date: string) => ComponentChildren
  onKeyDown?: KeyboardEventHandler<HTMLDivElement>
  onClickCapture?: MouseEventHandler<HTMLDivElement>
  /** Rendered after the grid, inside the root: a preview popover, for instance. */
  children?: ComponentChildren
}) => (
  <div class={classes.root} onKeyDown={onKeyDown} onClickCapture={onClickCapture}>
    <div class={classes.weekdays} aria-hidden="true">
      {dates.slice(0, 7).map(date => (
        <span key={date}>{weekdayLabel(date)}</span>
      ))}
    </div>
    <div
      id={id}
      class={classes.grid}
      style={{ gridTemplateRows: `repeat(${dates.length / 7}, minmax(${rowMinHeight}, 1fr))` }}
    >
      {dates.map((date, index) => (
        <section
          key={date}
          class={classes.day}
          data-date={date}
          data-outside={date.slice(0, 7) !== month.slice(0, 7) ? 'true' : undefined}
          data-today={date === today ? 'true' : undefined}
          aria-labelledby={`${id}-day-${date}`}
          {...dayAttributes?.(date)}
        >
          <header class={classes.dayHeader}>
            <h3 id={`${id}-day-${date}`}>
              <time
                dateTime={date}
                aria-label={dayLabel(date, index)}
                aria-current={date === today ? 'date' : undefined}
              >
                {Number(date.slice(8))}
              </time>
            </h3>
            {dayHeader?.(date)}
          </header>
          {renderDay(date)}
        </section>
      ))}
    </div>
    {children}
  </div>
)
