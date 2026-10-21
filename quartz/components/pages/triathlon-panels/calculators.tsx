import type { TriathlonRenderData } from '../../triathlon/render-data'
import { TRI_RACE_DISTANCES } from '../../../util/triathlon-calculator'
import { ShortcutHint } from '../../triathlon/shell/ShortcutHint'
import { TirePressure } from '../../triathlon/tools/TirePressure'
import { GearRatioCalculator } from './gear-ratios'

const CalcCopyButton = () => (
  <button
    class="tri-calc-copy"
    type="button"
    data-calc-tab-control="race"
    data-site-cursor-action
    aria-label="Copy embed link"
    title="Copy embed link"
  >
    <svg
      class="copy-icon"
      width="16"
      height="16"
      viewBox="-4 -4 24 24"
      fill="currentColor"
      aria-hidden="true"
      data-site-cursor-icon
    >
      <use href="#github-copy" />
    </svg>
    <svg
      class="check-icon"
      width="16"
      height="16"
      viewBox="-4 -4 24 24"
      fill="currentColor"
      aria-hidden="true"
      data-site-cursor-icon
    >
      <use href="#github-check" />
    </svg>
  </button>
)

export const CalcPanel = ({
  page,
  defaultDistance,
  renderData,
}: {
  page?: boolean
  defaultDistance?: unknown
  renderData?: TriathlonRenderData
}) => {
  const [defaultLabel, defaultSwim, defaultBike, defaultRun] =
    TRI_RACE_DISTANCES.find(([label]) => label === defaultDistance) ?? TRI_RACE_DISTANCES[1]

  return (
    <aside
      class={`tri-calc${page ? ' tri-calc--page' : ''}`}
      aria-hidden={page ? 'false' : 'true'}
      role={page ? 'region' : 'dialog'}
      aria-label={page ? 'calculators' : 'triathlon calculator'}
      data-swim={defaultSwim}
      data-bike={defaultBike}
      data-run={defaultRun}
      tabindex={-1}
      data-keyboard-scroll-scope
      data-keyboard-scroll
    >
      {!page && (
        <div class="tri-calc-bar">
          <span class="tri-calc-title">race pace</span>
          <button
            class="tri-calc-close tri-key-anchor"
            type="button"
            aria-label="Close"
            data-site-cursor-close
          >
            <span aria-hidden="true" data-site-cursor-icon>
              ×
            </span>
            <ShortcutHint>esc</ShortcutHint>
          </button>
        </div>
      )}
      {page && (
        <div class="tri-ana-bar">
          <div class="tri-calc-tabs" role="tablist" aria-label="calculators">
            <button
              id="tri-calc-tab-race"
              class="tri-calc-tab tri-calc-tab--on tri-key-anchor"
              type="button"
              role="tab"
              aria-selected="true"
              aria-controls="tri-calc-panel-race"
              aria-keyshortcuts="r"
              data-calc-tab="race"
            >
              race
              <ShortcutHint>r</ShortcutHint>
            </button>
            <button
              id="tri-calc-tab-gear-ratios"
              class="tri-calc-tab tri-key-anchor"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="tri-calc-panel-gear-ratios"
              aria-keyshortcuts="c"
              data-calc-tab="gear-ratios"
              tabindex={-1}
            >
              gear ratios
              <ShortcutHint>c</ShortcutHint>
            </button>
            <button
              id="tri-calc-tab-tire-pressure"
              class="tri-calc-tab tri-key-anchor"
              type="button"
              role="tab"
              aria-selected="false"
              aria-controls="tri-calc-panel-tire-pressure"
              aria-keyshortcuts="t"
              data-calc-tab="tire-pressure"
              tabindex={-1}
            >
              tire pressure
              <ShortcutHint>t</ShortcutHint>
            </button>
          </div>
        </div>
      )}
      <div
        id={page ? 'tri-calc-panel-race' : undefined}
        class="tri-calc-cell tri-calc-panel"
        role={page ? 'tabpanel' : undefined}
        aria-labelledby={page ? 'tri-calc-tab-race' : undefined}
        data-calc-panel="race"
      >
        <div class="tri-calc-presets">
          {TRI_RACE_DISTANCES.map(([label, s, b, r]) => (
            <button
              class={`tri-calc-preset${label === defaultLabel ? ' tri-calc-preset--on' : ''}`}
              type="button"
              data-swim={s}
              data-bike={b}
              data-run={r}
            >
              {label}
            </button>
          ))}
        </div>
        <div class="tri-calc-source" hidden>
          <div class="tri-calc-srcs" role="tablist" aria-label="pace source">
            <button
              class="tri-calc-src tri-calc-src--on"
              type="button"
              role="tab"
              aria-selected="true"
              data-src="avg"
            >
              average
            </button>
            <button
              class="tri-calc-src"
              type="button"
              role="tab"
              aria-selected="false"
              data-src="pred"
            >
              projected
            </button>
            <button
              class="tri-calc-src tri-calc-src--proj"
              type="button"
              role="tab"
              aria-selected="false"
              data-src="proj"
              data-i18n="projection"
              hidden
            >
              projection
            </button>
          </div>
          <CalcCopyButton />
        </div>
        <div class="tri-calc-box">
          <table class="tri-calc-io">
            <tbody>
              <tr>
                <th>swim</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="swim"
                    type="text"
                    value="2:00"
                    aria-label="swim pace"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">/100m</td>
                <td class="tri-calc-r" data-leg="swim">
                  <input
                    class="tri-calc-in tri-calc-legtime"
                    data-legtime="swim"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="swim time"
                    inputMode="numeric"
                  />
                </td>
              </tr>
              <tr>
                <th>T1</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="t1"
                    type="text"
                    value="5:00"
                    aria-label="T1 time"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">min</td>
                <td class="tri-calc-r" data-leg="t1">
                  —
                </td>
              </tr>
              <tr>
                <th>bike</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="bike"
                    type="text"
                    value="18"
                    aria-label="bike speed"
                    inputMode="decimal"
                  />
                </td>
                <td class="tri-calc-u" data-u="bike">
                  mph
                </td>
                <td class="tri-calc-r" data-leg="bike">
                  <input
                    class="tri-calc-in tri-calc-legtime"
                    data-legtime="bike"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="bike time"
                    inputMode="numeric"
                  />
                </td>
              </tr>
              <tr>
                <th>T2</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="t2"
                    type="text"
                    value="5:00"
                    aria-label="T2 time"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">min</td>
                <td class="tri-calc-r" data-leg="t2">
                  —
                </td>
              </tr>
              <tr>
                <th>run</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="run"
                    type="text"
                    value="9:00"
                    aria-label="run pace"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u" data-u="run">
                  /mi
                </td>
                <td class="tri-calc-r" data-leg="run">
                  <input
                    class="tri-calc-in tri-calc-legtime"
                    data-legtime="run"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="run time"
                    inputMode="numeric"
                  />
                </td>
              </tr>
              <tr class="tri-calc-total">
                <th>finish</th>
                <td />
                <td />
                <td class="tri-calc-r tri-calc-target-cell" data-leg="total">
                  <input
                    class="tri-calc-in tri-calc-target"
                    data-k="target"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="target finish time"
                    inputMode="numeric"
                  />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="tri-calc-proj" data-calc-tab-control="race" hidden>
        <div class="tri-calc-proj-zones" role="tablist" aria-label="heart rate zone" />
        <div class="tri-calc-proj-out" aria-live="polite" />
      </div>
      {page && (
        <div
          id="tri-calc-panel-gear-ratios"
          class="tri-calc-panel tri-calc-panel--tool"
          role="tabpanel"
          aria-labelledby="tri-calc-tab-gear-ratios"
          data-calc-panel="gear-ratios"
          hidden
        >
          <GearRatioCalculator />
        </div>
      )}
      {page && (
        <div
          id="tri-calc-panel-tire-pressure"
          class="tri-calc-panel tri-calc-panel--tool"
          role="tabpanel"
          aria-labelledby="tri-calc-tab-tire-pressure"
          data-calc-panel="tire-pressure"
          hidden
        >
          <TirePressure
            composition={renderData?.analytics.body.composition}
            weather={renderData?.weather}
          />
        </div>
      )}
    </aside>
  )
}
