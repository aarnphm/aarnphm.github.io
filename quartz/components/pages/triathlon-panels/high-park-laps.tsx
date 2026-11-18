import type { HighParkVisit } from '../../../util/high-park-loops'
import { PICK_SLOTS } from '../../../util/high-park-lap-compare'
import { highParkUnits, profileFields, profileModel, type LapSport } from './high-park-laps/fields'
import { LapChart, lapMetrics } from './high-park-laps/lap-chart'
import { LapTable } from './high-park-laps/lap-table'
import { Profile } from './high-park-laps/profile'

export * from './high-park-laps/fields'
export { LAP_CHART, lapChartScale, lapChartX, lapMetrics } from './high-park-laps/lap-chart'
export { profileLine, READ_WHO } from './high-park-laps/profile'

/** What the ride's laps reduce to, shared by the markup and the client that reads it. */
export const lapViewModel = (ride: Pick<HighParkVisit, 'lapDetail'>) => {
  const laps = ride.lapDetail
  return {
    /** The fastest lap's index. */
    best: laps.reduce((b, lap, i) => (lap.s < laps[b].s ? i : b), 0),
    lapKm: laps.reduce((sum, lap) => sum + lap.km, 0) / Math.max(1, laps.length),
  }
}

/** The averaged-lap mode's name: grade-adjusted pace on foot, a smoothed mean on the bike. */
export const baselineLabel = (sport: LapSport): string => (sport === 'run' ? 'GAP' : 'smoothed')

// Picks, the clear control and the baseline switch. The pick chips are pooled: the client shows
// one per picked lap and writes its number in place.
const Bar = ({ href, best, sport }: { href: string; best: number; sport: LapSport }) => (
  <div class="tri-hp-bar">
    <a class="internal tri-hp-open" href={href}>
      open activity
    </a>
    <span class="tri-hp-picks">
      <span class="tri-hp-pick" data-who="best">
        <i />
        {best + 1} fastest
      </span>
      {Array.from({ length: PICK_SLOTS }, (_, slot) => (
        <button
          key={slot}
          class="tri-hp-pick"
          type="button"
          data-who={slot}
          title="drop this lap"
          style="display:none"
        >
          <i />
          <span> </span>
        </button>
      ))}
      <button class="tri-hp-clear" type="button" style="display:none">
        clear
      </button>
    </span>
    <button class="tri-hp-mode" type="button" aria-pressed="false">
      {baselineLabel(sport)}
    </button>
  </div>
)

/**
 * The ride's drop-down: a link to the activity, laps compared side by side, per lap, as a table.
 * Static as rendered; fun/high-park/lap-select.ts adds picking, the hover readout and the baseline.
 */
export const LapView = ({
  ride,
  index,
  imperial = true,
  sport = 'bike',
}: {
  ride: Pick<HighParkVisit, 'lapDetail' | 'altitude' | 'href'>
  index?: number
  imperial?: boolean
  sport?: LapSport
}) => {
  const laps = ride.lapDetail
  if (laps.length === 0) return null
  const units = highParkUnits(imperial)
  const { best, lapKm } = lapViewModel(ride)
  return (
    <div class="tri-hp-lapview" data-ride={index} data-sport={sport}>
      <Bar href={ride.href} best={best} sport={sport} />
      {laps.length > 1 && (
        <section class="tri-hp-lapcharts" aria-label="lap by lap">
          <h3>
            lap by lap
            {/* The client names the lap under the pointer here. */}
            <span class="tri-hp-lapname"> </span>
          </h3>
          <div>
            {lapMetrics(units, sport).map(metric => (
              <LapChart key={metric.key} metric={metric} laps={laps} best={best} />
            ))}
          </div>
        </section>
      )}
      <section class="tri-hp-profiles" aria-label="per lap">
        <h3>per lap</h3>
        <div>
          {profileFields(units, sport).map(field => {
            const model = profileModel(field, laps, ride.altitude, lapKm, sport)
            return (
              model && (
                <Profile
                  key={field.key}
                  field={field}
                  model={model}
                  best={best}
                  altitude={ride.altitude}
                  lapKm={lapKm}
                  units={units}
                  sport={sport}
                />
              )
            )
          })}
        </div>
      </section>
      <LapTable laps={laps} best={best} units={units} sport={sport} />
    </div>
  )
}
