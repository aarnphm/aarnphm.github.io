import type { ComponentChildren } from 'preact'
import type { TriathlonRenderData } from '../../triathlon/render-data'
import { ROUTE_SPORTS, SPORT_ICON } from '../../../plugins/stores/strava'
import { MOUNTAIN_OUTLINE_PATH } from '../../../util/mountain-icon'
import { LAYERS_ICON } from '../../../util/triathlon-card'
import {
  ANALYTICS_CHAPTERS,
  analyticsPanelDefinition,
  DEFAULT_ANALYTICS_CHAPTER,
} from '../../triathlon/analytics/catalog'
import { AnalyticsServerPanel } from '../../triathlon/analytics/render'
import { CalendarPanel, CalendarSourceControls } from '../../triathlon/calendar/Calendar'
import { ShortcutHint } from '../../triathlon/shell/ShortcutHint'

const BACK_ICON = 'M19 12H5M11 6l-6 6 6 6'

const BackIcon = () => (
  <svg class="tri-ana-bar-back" viewBox="0 0 24 24" aria-hidden="true">
    <path d={BACK_ICON} />
  </svg>
)

const DISPATCH_ICON =
  'M189 375Q189 338 207 306.5Q225 275 256.5 257Q288 239 325 239H675Q712 239 743.5 257Q775 275 793 306.5Q811 338 811 375V775Q811 812 793 843.5Q775 875 743.5 893Q712 911 675 911H325Q288 911 256.5 893Q225 875 207 843.5Q189 812 189 775ZM261 375V775Q261 802 279.5 820.5Q298 839 325 839H675Q702 839 720.5 820.5Q739 802 739 775V375Q739 348 720.5 329.5Q702 311 675 311H325Q298 311 279.5 329.5Q261 348 261 375ZM411 275H339V100Q339 85 349.5 74.5Q360 64 375 64Q390 64 400.5 74.5Q411 85 411 100ZM661 275H589V150Q589 135 599.5 124.5Q610 114 625 114Q640 114 650.5 124.5Q661 135 661 150ZM375 539H625A36 36 0 0 1 625 611H375A36 36 0 0 1 375 539Z'

type TriPanelKind = 'analytics' | 'map' | 'calendar'

const ActivitySearchInput = ({ kind }: { kind: 'map' | 'compare' }) => (
  <input
    class={`tri-ana-search tri-${kind}-search`}
    type="search"
    placeholder="search (filter:bike|run|swim|walk, sort:distance|pace|cadence)"
    aria-label={kind === 'map' ? 'search routes' : 'search comparison activities'}
    aria-controls={kind === 'compare' ? 'tri-analytics-results' : undefined}
    autocomplete="off"
  />
)

interface TriPanelShellProps {
  kind: TriPanelKind
  id?: string
  panelClass?: string
  page?: boolean
  label: string
  barClass?: string
  bodyClass?: string
  controls?: ComponentChildren
  children: ComponentChildren
}

const TriPanelShell = ({
  kind,
  id,
  panelClass,
  page,
  label,
  barClass,
  bodyClass,
  controls,
  children,
}: TriPanelShellProps) => {
  const rootClass = `tri-${kind}`
  return (
    <>
      <div class={`${rootClass}-scrim`} aria-hidden="true" />
      <aside
        id={id}
        class={`${panelClass ?? rootClass}${page ? ` ${rootClass}--page` : ''}`}
        aria-hidden={page ? 'false' : 'true'}
        role="dialog"
        aria-label={label}
        tabindex={-1}
        data-keyboard-scroll-scope
      >
        <div class={`tri-ana-bar${barClass ? ` ${barClass}` : ''}`}>
          {controls}
          <button
            class={`tri-ana-close tri-key-anchor${kind === 'analytics' ? '' : ` tri-${kind}-close`}`}
            type="button"
            aria-label="Close"
            data-site-cursor-close
          >
            <span aria-hidden="true" data-site-cursor-icon>
              ×
            </span>
            {kind === 'analytics' && <BackIcon />}
            <ShortcutHint>esc</ShortcutHint>
          </button>
        </div>
        <div class={`tri-ana-body${bodyClass ? ` ${bodyClass}` : ''}`} data-keyboard-scroll>
          {children}
        </div>
      </aside>
    </>
  )
}

export const AnalyticsPanel = ({
  page,
  renderData,
}: {
  page?: boolean
  renderData?: TriathlonRenderData
}) => (
  <TriPanelShell
    kind="analytics"
    page={page}
    label="triathlon analytics"
    controls={
      <div class="tri-analytics-controls">
        <div
          class="tri-ana-chapters"
          role="tablist"
          aria-label="analytics chapters"
          data-i18n-aria-label="analytics chapters"
        >
          {ANALYTICS_CHAPTERS.map(chapter => (
            <button
              type="button"
              role="tab"
              id={`tri-ana-chapter-tab-${chapter.key}`}
              class="tri-ana-chapter-tab tri-key-anchor"
              data-analytics-chapter-tab={chapter.key}
              aria-controls={`tri-ana-chapter-${chapter.key}`}
              aria-selected={chapter.key === DEFAULT_ANALYTICS_CHAPTER ? 'true' : 'false'}
              aria-keyshortcuts={chapter.shortcut}
              tabindex={chapter.key === DEFAULT_ANALYTICS_CHAPTER ? 0 : -1}
            >
              <span data-i18n={chapter.key}>{chapter.key}</span>
              <ShortcutHint>{chapter.shortcut}</ShortcutHint>
            </button>
          ))}
        </div>
        <span class="tri-compare-picker-submit-wrap">
          <button
            class="tri-compare-picker-submit"
            type="button"
            disabled
            aria-label="compare selected"
            data-i18n-aria-label="compare selected"
            aria-describedby="tri-compare-picker-submit-help"
            aria-keyshortcuts="Shift+Enter"
            data-site-cursor-action
          >
            <svg
              class="tri-compare-picker-icon"
              viewBox="0 0 24 24"
              aria-hidden="true"
              data-site-cursor-icon
            >
              <path d="M12 4.5 19.5 12 12 19.5 4.5 12Z" />
            </svg>
          </button>
          <span
            id="tri-compare-picker-submit-help"
            class="tri-compare-picker-tooltip"
            role="tooltip"
          >
            <span data-i18n="compare activities">compare activities</span>
          </span>
        </span>
        <button
          class="tri-ana-compare-toggle tri-key-anchor"
          type="button"
          aria-pressed="false"
          aria-label="compare activities"
          aria-controls="tri-analytics-results"
          aria-keyshortcuts="c"
          data-i18n-aria-label="compare activities"
          data-site-cursor-action
        >
          <svg class="tri-ana-compare-icon" viewBox="0 0 1000 1000" aria-hidden="true">
            <path d={DISPATCH_ICON} />
          </svg>
          <BackIcon />
          <ShortcutHint>c</ShortcutHint>
        </button>
      </div>
    }
  >
    <div class="tri-compare-search-wrap">
      <ActivitySearchInput kind="compare" />
    </div>
    <div id="tri-analytics-results" class="tri-ana-results" aria-hidden="true" />
    <div id="tri-analytics-detail" class="tri-ana-detail" aria-hidden="true" />
    {ANALYTICS_CHAPTERS.map(chapter => (
      <section
        id={`tri-ana-chapter-${chapter.key}`}
        class="tri-ana-chapter"
        role="tabpanel"
        aria-labelledby={`tri-ana-chapter-tab-${chapter.key}`}
        data-analytics-chapter={chapter.key}
        hidden={chapter.key !== DEFAULT_ANALYTICS_CHAPTER}
      >
        {chapter.panels.map(key => {
          const definition = analyticsPanelDefinition(key)
          return (
            <div class="tri-ana-block" data-chart={key}>
              {page && renderData && definition && (
                <AnalyticsServerPanel definition={definition} data={renderData.analytics} />
              )}
            </div>
          )
        })}
      </section>
    ))}
  </TriPanelShell>
)

export const MapPanel = ({ page }: { page?: boolean }) => (
  <TriPanelShell
    kind="map"
    page={page}
    label="triathlon route maps"
    barClass="tri-map-bar"
    bodyClass="tri-map-body"
    controls={
      <div class="tri-map-search-wrap tri-key-anchor tri-key-search">
        <ActivitySearchInput kind="map" />
        <ShortcutHint>/</ShortcutHint>
        <div class="tri-ana-results tri-map-results" aria-hidden="true" />
      </div>
    }
  >
    <div class="tri-map-pane">
      <div class="tri-map-canvas" />
      <div class="tri-map-overlay">
        <div class="tri-map-modes" role="group" aria-label="map overlay metric">
          <button
            class="tri-map-mode"
            type="button"
            data-mode="heat"
            aria-pressed="true"
            data-i18n="heat"
          >
            heat
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="w"
            aria-pressed="false"
            data-i18n="power"
          >
            power
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="hr"
            aria-pressed="false"
            data-i18n="hr"
          >
            hr
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="cad"
            aria-pressed="false"
            data-i18n="cadence"
          >
            cadence
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="spd"
            aria-pressed="false"
            data-i18n="speed"
          >
            speed
          </button>
        </div>
        <div class="tri-map-legend tri-map-overlay-legend">
          <span class="tri-map-legend-bar" />
          <span class="tri-map-legend-ends">
            <span class="tri-map-legend-lo" />
            <span class="tri-map-legend-hi" />
          </span>
        </div>
      </div>
      <div class="tri-map-side" role="group" aria-label="map controls">
        <button
          class="tri-map-side-fold"
          type="button"
          aria-expanded="true"
          aria-label="Collapse map controls"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
        <div class="tri-map-side-body">
          {ROUTE_SPORTS.map(sport => (
            <button
              class="tri-map-sport"
              type="button"
              data-sport={sport}
              aria-pressed="true"
              aria-label={sport}
              title={sport}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                {SPORT_ICON[sport].map(d => (
                  <path d={d} />
                ))}
              </svg>
            </button>
          ))}
          <span class="tri-map-side-rule" />
          <button
            class="tri-map-elevation"
            type="button"
            aria-pressed="false"
            aria-label="Elevation contours and hill shading"
            title="Elevation contours and hill shading"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d={MOUNTAIN_OUTLINE_PATH} />
            </svg>
          </button>
          <button
            class="tri-map-3d"
            type="button"
            aria-pressed="false"
            aria-label="3D terrain and buildings"
            title="3D terrain and buildings"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3 20 7.5 12 12 4 7.5 12 3Z" />
              <path d="M4 7.5v9L12 21v-9" />
              <path d="M20 7.5v9L12 21" />
            </svg>
          </button>
          <button
            class="tri-map-style"
            type="button"
            aria-pressed="false"
            aria-label="satellite"
            title="satellite"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              {LAYERS_ICON.map(d => (
                <path d={d} />
              ))}
            </svg>
          </button>
        </div>
      </div>
      <div class="tri-map-selection" aria-hidden="true" />
      <div class="tri-map-tip" aria-hidden="true" />
    </div>
    <div
      class="tri-ana-detail tri-map-detail tri-map-sidebar"
      aria-hidden="true"
      data-keyboard-scroll
    />
  </TriPanelShell>
)

export const CalendarOverlay = ({ renderData }: { renderData?: TriathlonRenderData }) => (
  <TriPanelShell
    kind="calendar"
    id="tri-calendar-panel"
    panelClass="tri-calendar-panel"
    label="triathlon calendar"
    controls={<CalendarSourceControls id="tri-calendar" />}
  >
    <CalendarPanel renderData={renderData} id="tri-calendar" panel />
  </TriPanelShell>
)
