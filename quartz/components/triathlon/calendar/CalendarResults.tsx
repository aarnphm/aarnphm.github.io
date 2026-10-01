import type { TriathlonCalendarResults } from '../../../util/triathlon-calendar'
import { TRIATHLON_CALENDAR_RESULT_SEGMENTS } from '../../../util/triathlon-calendar'

export const CalendarResults = ({ results }: { results: TriathlonCalendarResults }) => (
  <section class="tri-calendar-card-results" data-card-detail>
    <h3 data-i18n="race results">race results</h3>
    <dl class="tri-calendar-card-stats">
      {TRIATHLON_CALENDAR_RESULT_SEGMENTS.map(segment => {
        const time = results[segment]
        return (
          time !== null && (
            <div data-result-segment={segment}>
              <dt data-i18n={segment}>{segment}</dt>
              <dd>{time}</dd>
            </div>
          )
        )
      })}
    </dl>
  </section>
)
