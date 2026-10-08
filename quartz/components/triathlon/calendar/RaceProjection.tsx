import type { TriathlonCalendarEvent } from '../../../util/triathlon-calendar'
import { raceProjectionSpec } from '../../../util/race-projection'

export const RaceProjection = ({ event }: { event: TriathlonCalendarEvent }) => {
  const spec = raceProjectionSpec(event)
  const inputId = (field: string): string => `race-projection-${event.id}-${field}`
  return (
    <section
      class="tri-calendar-race-projection"
      aria-label="race projection"
      data-i18n-aria-label="race projection"
      data-series={event.series}
      data-race-projection={JSON.stringify(spec)}
    >
      <h3>
        <span data-i18n="race projection">race projection</span> · {event.name}
      </h3>
      <div class="tri-calendar-projection-overview">
        <div class="tri-calendar-race-prediction" data-race-projection-output>
          <p data-i18n="loading training model">loading training model</p>
        </div>
        <table
          class="tri-calendar-projection-table tri-calendar-scenario"
          aria-label="race projection scenario"
          data-i18n-aria-label="race projection scenario"
        >
          <tbody>
            <tr>
              <th scope="row">
                <label for={inputId('weeklyLoad')} data-i18n="weekly training TSS">
                  weekly training TSS
                </label>
              </th>
              <td>
                <input
                  id={inputId('weeklyLoad')}
                  type="number"
                  min="0"
                  max="3000"
                  step="1"
                  data-race-scenario="weeklyLoad"
                  disabled
                />
              </td>
            </tr>
            <tr>
              <th scope="row">
                <label for={inputId('taperDays')} data-i18n="taper days">
                  taper days
                </label>
              </th>
              <td>
                <input
                  id={inputId('taperDays')}
                  type="number"
                  min="0"
                  max="21"
                  step="1"
                  value="7"
                  data-race-scenario="taperDays"
                />
              </td>
            </tr>
            <tr>
              <th scope="row">
                <label for={inputId('paceGainPct')} data-i18n="pace gain goal %">
                  pace gain goal %
                </label>
              </th>
              <td>
                <input
                  id={inputId('paceGainPct')}
                  type="number"
                  min="-10"
                  max="10"
                  step="1"
                  value="0"
                  data-race-scenario="paceGainPct"
                />
              </td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={2}>
                <button type="button" data-race-scenario-reset data-i18n="reset scenario">
                  reset scenario
                </button>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      <details class="tri-calendar-race-assumptions">
        <summary data-i18n="projection assumptions">projection assumptions</summary>
        <p
          class="tri-calendar-projection-basis"
          data-race-projection-basis
          role="status"
          data-i18n="loading training model"
        >
          loading training model
        </p>
        <div data-race-projection-fitness />
        <table class="tri-calendar-projection-table tri-calendar-race-inputs">
          <tbody>
            {spec.legs.map(leg => (
              <tr key={leg.sport}>
                <th scope="row">
                  <label for={inputId(leg.sport)}>
                    <span data-i18n={leg.sport}>{leg.sport}</span> ·{' '}
                    {leg.distanceKm.toLocaleString('en-CA', { maximumFractionDigits: 3 })} km · IF
                  </label>
                  <small>
                    {leg.elevationM} m ↑{leg.referenceYear ? ` · ${leg.referenceYear}` : ''} ·{' '}
                    <span data-i18n={`${leg.distanceSource} distance`}>
                      {leg.distanceSource} distance
                    </span>
                  </small>
                </th>
                <td>
                  <input
                    id={inputId(leg.sport)}
                    type="number"
                    min="0.4"
                    max="1.15"
                    step="0.01"
                    value={leg.intensity}
                    data-race-intensity={leg.sport}
                  />
                </td>
              </tr>
            ))}
            {event.kind === 'hyrox' && (
              <tr>
                <th scope="row">
                  <label for={inputId('division')} data-i18n="HYROX division">
                    HYROX division
                  </label>
                </th>
                <td>
                  <select id={inputId('division')} data-race-division>
                    <option value="" data-i18n="division pending">
                      division pending
                    </option>
                    <option value="singles-open">Singles Open</option>
                    <option value="singles-pro">Singles Pro</option>
                    <option value="doubles-open">Doubles Open</option>
                    <option value="doubles-pro">Doubles Pro</option>
                  </select>
                </td>
              </tr>
            )}
            <tr class="tri-calendar-race-extra">
              <th scope="row">
                <label
                  for={inputId('extra')}
                  data-i18n={
                    event.kind === 'hyrox' ? 'stations + Roxzone min' : 'transitions / stops min'
                  }
                >
                  {event.kind === 'hyrox' ? 'stations + Roxzone min' : 'transitions / stops min'}
                </label>
              </th>
              <td>
                <input
                  id={inputId('extra')}
                  type="number"
                  min="0"
                  max="180"
                  step="1"
                  value={spec.extraMinutes ?? ''}
                  data-race-extra
                  disabled={event.kind === 'hyrox'}
                />
              </td>
            </tr>
            {spec.legs.some(leg => leg.sport === 'run') && (
              <tr>
                <th scope="row">
                  <label for={inputId('runPenalty')} data-i18n="run fatigue penalty %">
                    run fatigue penalty %
                  </label>
                </th>
                <td>
                  <input
                    id={inputId('runPenalty')}
                    type="number"
                    min="0"
                    max="50"
                    step="1"
                    value={spec.runPenaltyPct}
                    data-race-run-penalty
                  />
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <p data-i18n="Training load changes fitness and fatigue inputs. Pace gain is an explicit goal. Taper halves the assumed load. Each race is projected independently.">
          Training load changes fitness and fatigue inputs. Pace gain is an explicit goal. Taper
          halves the assumed load. Each race is projected independently.
        </p>
        <p data-i18n="IF sets the TSS estimate; the pace model supplies time. Conditions are unspecified. The range is a planning spread, not a calibrated race interval.">
          IF sets the TSS estimate; the pace model supplies time. Conditions are unspecified. The
          range is a planning spread, not a calibrated race interval.
        </p>
        <p data-i18n="Optimistic case: scenario time −10%. Uncertainty widens the slower case.">
          Optimistic case: scenario time −10%. Uncertainty widens the slower case.
        </p>
        {event.kind === 'hyrox' && (
          <p data-i18n="Enter a benchmark for this division. Changing division clears station time. HYROX TSS is a duration/intensity proxy and does not measure muscular load.">
            Enter a benchmark for this division. Changing division clears station time. HYROX TSS is
            a duration/intensity proxy and does not measure muscular load.
          </p>
        )}
        <p class="tri-calendar-projection-method">
          <span data-i18n="estimated race TSS = hours × IF² × 100">
            estimated race TSS = hours × IF² × 100
          </span>{' '}
          ·{' '}
          <a
            href="https://www.trainingpeaks.com/learn/articles/estimating-training-stress-score-tss/"
            target="_blank"
            rel="noopener noreferrer"
            data-no-popover
          >
            TrainingPeaks
          </a>
          {event.kind === 'hyrox' && (
            <>
              {' '}
              ·{' '}
              <a
                href="https://hyrox.com/wp-content/uploads/2025/07/25_26_HYROX_RulebookSingles_EN.pdf"
                target="_blank"
                rel="noopener noreferrer"
                data-no-popover
              >
                HYROX 26/27
              </a>
            </>
          )}
        </p>
      </details>
    </section>
  )
}
