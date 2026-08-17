import type { BodyCompositionDay } from '../../../plugins/stores/analytics'
import type { WeatherSnapshot } from '../../../plugins/stores/weather'
import {
  calculateTirePressure,
  DEFAULT_TIRE_PRESSURE_SELECTION,
  formatTirePressurePsi,
  latestMorningBodyWeight,
  PIRELLI_PRESSURE_SOURCE_URL,
  TIRE_PRESSURE_BIKES,
  TIRE_PRESSURE_MEASURED_WIDTH_MM,
  TIRE_PRESSURE_SOURCE_URL,
  TIRE_PRESSURE_SURFACES,
  TIRE_PRESSURE_TIRES,
  TIRE_PRESSURE_WHEELS,
} from '../../../util/triathlon-tire-pressure'

interface TirePressureProps {
  composition?: readonly BodyCompositionDay[]
  weather?: WeatherSnapshot | null
}

const weatherCondition = (weather: WeatherSnapshot): string => {
  const value = weather.conditionCode ?? weather.precipitationType
  if (!value) return 'conditions unavailable'
  return value
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replaceAll('_', ' ')
    .toLowerCase()
}

export const TirePressure = ({ composition = [], weather = null }: TirePressureProps) => {
  const morningWeight = latestMorningBodyWeight(composition)
  const recommendation = morningWeight
    ? calculateTirePressure(morningWeight.kg, DEFAULT_TIRE_PRESSURE_SELECTION)
    : null
  return (
    <section
      id="tire-pressure"
      class="tri-pressure"
      aria-label="tire pressure calculator"
      data-rider-kg={morningWeight?.kg}
      data-weight-date={morningWeight?.date}
      data-bike={DEFAULT_TIRE_PRESSURE_SELECTION.bike}
      data-wheel={DEFAULT_TIRE_PRESSURE_SELECTION.wheel}
      data-tire={DEFAULT_TIRE_PRESSURE_SELECTION.tire}
      data-surface={DEFAULT_TIRE_PRESSURE_SELECTION.surface}
      data-speed-mph={DEFAULT_TIRE_PRESSURE_SELECTION.speedMph}
    >
      <div class="tri-pressure-head">
        <span data-i18n="tire pressure">tire pressure</span>
        <span class="tri-pressure-date">
          {morningWeight ? (
            <>
              <time datetime={morningWeight.date}>{morningWeight.date}</time>
              <span aria-hidden="true"> · </span>
              <span>{morningWeight.kg.toFixed(2)} kg</span>
              <span aria-hidden="true"> · </span>
              <span data-i18n="Garmin morning">Garmin morning</span>
            </>
          ) : (
            <span data-i18n="morning weight unavailable">morning weight unavailable</span>
          )}
        </span>
      </div>
      <div class="tri-pressure-result" aria-live="polite">
        <div class="tri-pressure-axle">
          <span class="tri-pressure-axle-label" data-i18n="front">
            front
          </span>
          <span class="tri-pressure-value-wrap">
            <output class="tri-pressure-value" data-pressure-output="front">
              {recommendation ? formatTirePressurePsi(recommendation.frontPsi) : '—'}
            </output>
            <span class="tri-pressure-unit">PSI</span>
          </span>
        </div>
        <div class="tri-pressure-axle">
          <span class="tri-pressure-axle-label" data-i18n="rear">
            rear
          </span>
          <span class="tri-pressure-value-wrap">
            <output class="tri-pressure-value" data-pressure-output="rear">
              {recommendation ? formatTirePressurePsi(recommendation.rearPsi) : '—'}
            </output>
            <span class="tri-pressure-unit">PSI</span>
          </span>
        </div>
        <span class="tri-pressure-system" data-pressure-system>
          {recommendation
            ? `${recommendation.riderKg.toFixed(1)} + ${recommendation.bikeKg.toFixed(1)} = ${recommendation.systemKg.toFixed(1)} kg system`
            : 'add a morning body-composition measurement'}
        </span>
      </div>
      <div class="tri-pressure-controls">
        <fieldset class="tri-pressure-field" data-pressure-group="bike">
          <legend data-i18n="bike">bike</legend>
          <div class="tri-pressure-options">
            {TIRE_PRESSURE_BIKES.map(bike => (
              <label class="tri-pressure-option">
                <input
                  type="radio"
                  name="tri-pressure-bike"
                  value={bike.id}
                  data-pressure-field="bike"
                  checked={bike.id === DEFAULT_TIRE_PRESSURE_SELECTION.bike}
                />
                <span>{bike.label}</span>
                <small>{bike.massLb} lb</small>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset class="tri-pressure-field" data-pressure-group="wheel">
          <legend data-i18n="wheelset">wheelset</legend>
          <div class="tri-pressure-options">
            {TIRE_PRESSURE_WHEELS.map(wheel => (
              <label class="tri-pressure-option">
                <input
                  type="radio"
                  name="tri-pressure-wheel"
                  value={wheel.id}
                  data-pressure-field="wheel"
                  checked={wheel.id === DEFAULT_TIRE_PRESSURE_SELECTION.wheel}
                />
                <span>{wheel.label}</span>
                <small>
                  {wheel.frontInnerWidthMm === wheel.rearInnerWidthMm
                    ? `${wheel.frontInnerWidthMm} mm internal`
                    : `${wheel.frontInnerWidthMm}/${wheel.rearInnerWidthMm} mm internal`}
                </small>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset class="tri-pressure-field" data-pressure-group="surface">
          <legend data-i18n="surface">surface</legend>
          <div class="tri-pressure-options tri-pressure-options--surface">
            {TIRE_PRESSURE_SURFACES.map(surface => (
              <label class="tri-pressure-option">
                <input
                  type="radio"
                  name="tri-pressure-surface"
                  value={surface.id}
                  data-pressure-field="surface"
                  checked={surface.id === DEFAULT_TIRE_PRESSURE_SELECTION.surface}
                  aria-label={`${surface.label}, SILCA coefficient ${surface.coefficient}, ${surface.note}`}
                />
                <span>{surface.label}</span>
                <span class="tri-pressure-surface-tip" aria-hidden="true">
                  <strong>{surface.coefficient}</strong>
                  <span>{surface.note}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset class="tri-pressure-field" data-pressure-group="tire">
          <legend data-i18n="tire setup">tire setup</legend>
          <div class="tri-pressure-options">
            {TIRE_PRESSURE_TIRES.map(tire => (
              <label class="tri-pressure-option">
                <input
                  type="radio"
                  name="tri-pressure-tire"
                  value={tire.id}
                  data-pressure-field="tire"
                  checked={tire.id === DEFAULT_TIRE_PRESSURE_SELECTION.tire}
                />
                <span>{tire.label}</span>
                <small>{tire.detail}</small>
              </label>
            ))}
          </div>
        </fieldset>
        <label class="tri-pressure-speed">
          <span data-i18n="average speed">average speed</span>
          <span class="tri-pressure-speed-input">
            <input
              type="text"
              value={DEFAULT_TIRE_PRESSURE_SELECTION.speedMph}
              data-pressure-field="speed"
              inputMode="decimal"
              aria-label="average speed in miles per hour"
            />
            <span>mph</span>
          </span>
        </label>
      </div>
      <div class="tri-pressure-spec">
        <span data-pressure-tire>Pirelli P Zero Race SL-R + P Zero TPU tube</span>
        <span>
          {TIRE_PRESSURE_MEASURED_WIDTH_MM} mm measured · <span data-pressure-diameter>622</span> mm
          BSD · dry
        </span>
        <span data-pressure-rim>22 mm internal</span>
      </div>
      <aside class="tri-pressure-weather" aria-label="weather pressure guidance">
        <div class="tri-pressure-weather-head">
          <span data-i18n="ride conditions">ride conditions</span>
          {weather ? (
            <time datetime={weather.forecastStart}>
              WeatherKit · {weather.forecastStart.replace('T', ' ').slice(0, 16)} UTC
            </time>
          ) : (
            <span data-i18n="WeatherKit forecast unavailable">WeatherKit forecast unavailable</span>
          )}
        </div>
        {weather && (
          <p class="tri-pressure-weather-current">
            <span>
              {weather.temperatureC == null
                ? 'temperature unavailable'
                : `${weather.temperatureC.toFixed(1)} °C`}
            </span>
            <span>{weatherCondition(weather)}</span>
            {weather.precipitationChance != null && (
              <span>{Math.round(weather.precipitationChance * 100)}% precipitation</span>
            )}
          </p>
        )}
        <div class="tri-pressure-weather-offsets">
          <span>
            <strong data-i18n="mixed">mixed</strong>
            <span>−3 PSI</span>
          </span>
          <span>
            <strong data-i18n="wet">wet</strong>
            <span>−8 PSI</span>
          </span>
        </div>
        <p class="tri-pressure-weather-note">
          <span>
            Temperature changes pressure by about 3 PSI per 10 °C between inflation and riding
            conditions. Altitude changes it by about 1.5 PSI per 1,000 m. The values above remain
            the dry SILCA baseline; inspect the road and respect the lower tire/rim limit.
          </span>
        </p>
      </aside>
      <p class="tri-pressure-warning" data-pressure-warning hidden>
        Reserve specifies 29 mm as its minimum recommended tire width. This 28 mm setup sits below
        that published range.
      </p>
      <p class="tri-pressure-note">
        <span>Fastest-pressure estimate from the </span>
        <a href={TIRE_PRESSURE_SOURCE_URL} target="_blank" rel="noopener noreferrer">
          SILCA field model
        </a>
        <span>. Confirm the lower tire/rim limit with </span>
        <a href={PIRELLI_PRESSURE_SOURCE_URL} target="_blank" rel="noopener noreferrer">
          Pirelli
        </a>
        <span>.</span>
      </p>
    </section>
  )
}
