import { InlineMath } from '../../../util/math-text'
import {
  CERAMICSPEED_CROSS_CHAIN_RESEARCH,
  CERAMICSPEED_TEST_CADENCE_RPM,
  CERAMICSPEED_TEST_CHAINSTAY_MM,
  CERAMICSPEED_TEST_OUTPUT_WATTS,
  DEFAULT_GEAR_CASSETTE,
  DEFAULT_GEAR_CHAINRING_PRESET,
  formatGearEfficiencyDeltaPercent,
  GEAR_CASSETTE_PRESET_GROUPS,
  GEAR_CHAINRING_PRESETS,
  gearRatioMatrix,
  type GearCassettePreset,
} from '../../../util/triathlon-gear-ratio'

const DEFAULT_GEAR_CHAINRINGS = DEFAULT_GEAR_CHAINRING_PRESET.chainrings

const gearEfficiencyLevel = (crossChainLossWatts: number): string =>
  `${(8 + Math.min(crossChainLossWatts / 2.5, 1) * 32).toFixed(1)}%`

const GearRatioTable = ({
  chainrings,
  cassette,
}: {
  chainrings: readonly number[]
  cassette: GearCassettePreset
}) => {
  const matrix = gearRatioMatrix(chainrings, cassette.cogs)
  if (!matrix) return null
  return (
    <table class="tri-ratio-table" aria-label="gear ratio chart" data-i18n-aria-label="gear ratios">
      <caption>
        <InlineMath className="tri-math" tex="\Delta\eta" /> est. vs. ideal · CeramicSpeed{' '}
        <span class="tri-ratio-source-links" aria-label="CeramicSpeed research sources">
          {CERAMICSPEED_CROSS_CHAIN_RESEARCH.sources.map(source => (
            <a
              class="tri-ratio-source-link"
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              title={source.title}
              aria-label={`source ${source.id}: ${source.title}, opens in new tab`}
            >
              [{source.id}]
            </a>
          ))}
        </span>{' '}
        · {CERAMICSPEED_TEST_OUTPUT_WATTS} W · {CERAMICSPEED_TEST_CADENCE_RPM} rpm ·{' '}
        {CERAMICSPEED_TEST_CHAINSTAY_MM} mm chainstay
      </caption>
      <thead>
        <tr>
          <th scope="col" aria-label="chainring and cassette teeth">
            T
          </th>
          {cassette.cogs.map(cog => (
            <th scope="col">{cog}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.rows.map((row, rowIndex) => (
          <>
            <tr class={`tri-ratio-row tri-ratio-row--${rowIndex + 1}`}>
              <th scope="row" aria-label={`${row.chainring} tooth chainring`}>
                {row.chainring}
              </th>
              {row.cells.map(cell => (
                <td
                  data-ratio-chainring={row.chainring}
                  data-ratio-cog={cell.cog}
                  data-ratio-value={cell.ratio.toFixed(2)}
                  title={`${row.chainring}T ÷ ${cell.cog}T = ${cell.ratio.toFixed(2)}`}
                  style={`--tri-ratio-level:${(18 + cell.level * 52).toFixed(1)}%`}
                >
                  {cell.ratio.toFixed(2)}
                </td>
              ))}
            </tr>
            <tr class={`tri-ratio-efficiency-row tri-ratio-row--${rowIndex + 1}`}>
              <th
                scope="row"
                aria-label={`estimated CeramicSpeed drivetrain efficiency difference from ideal for ${row.chainring} tooth chainring`}
              >
                <InlineMath className="tri-math" tex="\Delta\eta" />
              </th>
              {row.cells.map((cell, cellIndex) => {
                const efficiency = cell.drivetrainEfficiency.toFixed(3)
                const efficiencyDelta = formatGearEfficiencyDeltaPercent(
                  cell.drivetrainEfficiency,
                  3,
                )
                const visibleEfficiencyDelta = formatGearEfficiencyDeltaPercent(
                  cell.drivetrainEfficiency,
                  2,
                )
                const compactEfficiencyDelta = formatGearEfficiencyDeltaPercent(
                  cell.drivetrainEfficiency,
                  1,
                )
                return (
                  <td
                    data-efficiency-chainring={row.chainring}
                    data-efficiency-cog={cell.cog}
                    data-efficiency-value={efficiency}
                    data-efficiency-delta={efficiencyDelta}
                    data-loss-watts={cell.drivetrainLossWatts.toFixed(2)}
                    data-cross-chain-loss-watts={cell.crossChainLossWatts.toFixed(2)}
                    aria-label={`${visibleEfficiencyDelta}% estimated drivetrain efficiency difference from ideal; ${cell.drivetrainEfficiency.toFixed(2)}% estimated drivetrain efficiency; ${cell.drivetrainLossWatts.toFixed(2)} watts drivetrain loss; ${cell.crossChainLossWatts.toFixed(2)} watts cross-chain loss`}
                    tabindex={rowIndex === 0 && cellIndex === 0 ? 0 : -1}
                    style={`--tri-efficiency-level:${gearEfficiencyLevel(cell.crossChainLossWatts)}`}
                  >
                    <span class="tri-ratio-efficiency-value tri-ratio-efficiency-value--full">
                      <InlineMath className="tri-math" tex={`${visibleEfficiencyDelta}\\%`} />
                    </span>
                    <span class="tri-ratio-efficiency-value tri-ratio-efficiency-value--compact">
                      <InlineMath className="tri-math" tex={`${compactEfficiencyDelta}\\%`} />
                    </span>
                  </td>
                )
              })}
            </tr>
          </>
        ))}
      </tbody>
    </table>
  )
}

export const GearRatioCalculator = () => {
  const matrix = gearRatioMatrix(DEFAULT_GEAR_CHAINRINGS, DEFAULT_GEAR_CASSETTE.cogs)
  return (
    <section
      class="tri-ratio"
      aria-label="gear ratio calculator"
      data-i18n-aria-label="gear ratios"
    >
      <div class="tri-ratio-head">
        <span data-i18n="gear ratios">gear ratios</span>
        <output class="tri-ratio-range" aria-live="polite">
          {matrix ? `${matrix.minimum.toFixed(2)}–${matrix.maximum.toFixed(2)}` : ''}
        </output>
      </div>
      <div class="tri-ratio-controls">
        <div class="tri-ratio-chainring-presets">
          <span id="tri-ratio-chainring-preset-label" data-i18n="presets">
            presets
          </span>
          <div class="tri-ratio-chainring-preset-picker">
            <button
              class="tri-ratio-chainring-preset-trigger"
              type="button"
              aria-labelledby="tri-ratio-chainring-preset-label tri-ratio-chainring-preset-value"
              aria-haspopup="listbox"
              aria-expanded="false"
              aria-controls="tri-ratio-chainring-preset-menu"
            >
              <span id="tri-ratio-chainring-preset-value" class="tri-ratio-chainring-preset-value">
                {DEFAULT_GEAR_CHAINRING_PRESET.label}
              </span>
              <svg
                class="tri-ratio-chainring-preset-chevron"
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
              id="tri-ratio-chainring-preset-menu"
              class="tri-ratio-chainring-preset-menu"
              role="listbox"
              aria-labelledby="tri-ratio-chainring-preset-label"
              hidden
            >
              {GEAR_CHAINRING_PRESETS.map(preset => (
                <button
                  class="tri-ratio-chainring-preset-option"
                  type="button"
                  role="option"
                  aria-selected={preset.id === DEFAULT_GEAR_CHAINRING_PRESET.id}
                  data-chainring-preset-id={preset.id}
                >
                  <span class="tri-ratio-chainring-preset-check" aria-hidden="true">
                    ✓
                  </span>
                  <span class="tri-ratio-chainring-preset-option-value">{preset.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
        <fieldset class="tri-ratio-rings">
          <legend data-i18n="chainrings">chainrings</legend>
          <div class="tri-ratio-ring-inputs">
            {DEFAULT_GEAR_CHAINRINGS.map((chainring, index) => (
              <label class="tri-ratio-ring" data-ratio-ring={index + 1}>
                <input
                  class="tri-ratio-ring-input"
                  type="number"
                  value={chainring}
                  min="24"
                  max="64"
                  step="1"
                  inputMode="numeric"
                  aria-label={`chainring ${index + 1} teeth`}
                />
                <span aria-hidden="true">T</span>
              </label>
            ))}
          </div>
        </fieldset>
        <div class="tri-ratio-cassette">
          <span id="tri-ratio-cassette-label" data-i18n="cassette">
            cassette
          </span>
          <div class="tri-ratio-cassette-picker">
            <button
              class="tri-ratio-cassette-trigger"
              type="button"
              aria-labelledby="tri-ratio-cassette-label tri-ratio-cassette-value"
              aria-haspopup="listbox"
              aria-expanded="false"
              aria-controls="tri-ratio-cassette-menu"
            >
              <span id="tri-ratio-cassette-value" class="tri-ratio-cassette-value">
                {DEFAULT_GEAR_CASSETTE.label}
              </span>
              <svg
                class="tri-ratio-cassette-chevron"
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
              id="tri-ratio-cassette-menu"
              class="tri-ratio-cassette-menu"
              role="listbox"
              aria-labelledby="tri-ratio-cassette-label"
              hidden
            >
              {GEAR_CASSETTE_PRESET_GROUPS.map(group => (
                <div class="tri-ratio-cassette-group" role="group" aria-label={group.label}>
                  <span class="tri-ratio-cassette-group-label" aria-hidden="true">
                    {group.label}
                  </span>
                  {group.presets.map(preset => (
                    <button
                      class="tri-ratio-cassette-option"
                      type="button"
                      role="option"
                      aria-selected={preset.id === DEFAULT_GEAR_CASSETTE.id}
                      data-cassette-id={preset.id}
                    >
                      <span class="tri-ratio-cassette-check" aria-hidden="true">
                        ✓
                      </span>
                      <span class="tri-ratio-cassette-option-value">{preset.label}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div class="tri-ratio-layout" role="group" aria-label="chainring count">
          <button
            class="tri-ratio-layout-btn tri-ratio-layout-btn--on"
            type="button"
            data-ratio-layout="2"
            aria-pressed="true"
          >
            2×
          </button>
          <button
            class="tri-ratio-layout-btn"
            type="button"
            data-ratio-layout="1"
            aria-pressed="false"
          >
            1×
          </button>
        </div>
      </div>
      <div class="tri-ratio-chart-scroll">
        <div class="tri-ratio-chart">
          <GearRatioTable chainrings={DEFAULT_GEAR_CHAINRINGS} cassette={DEFAULT_GEAR_CASSETTE} />
        </div>
      </div>
    </section>
  )
}
