import type { ComponentChildren, JSX } from 'preact'
import type { TrainingPlan } from '../../../plugins/stores/training'
import type { TriathlonRenderData } from '../../triathlon/render-data'
import { ROUTE_SPORTS, SPORT_ICON } from '../../../plugins/stores/strava'
import { MOUNTAIN_OUTLINE_PATH } from '../../../util/mountain-icon'
import { LAYERS_ICON } from '../../../util/triathlon-card'
import { ANALYTICS_CATALOG } from '../../triathlon/analytics/catalog'
import { AnalyticsServerPanel } from '../../triathlon/analytics/render'
import { CalendarPanel, CalendarSourceControls } from '../../triathlon/calendar/Calendar'
import { ShortcutHint } from '../../triathlon/shell/ShortcutHint'
import { deriveTrainingDocument, type TrainingTreeNode } from '../../triathlon/training/tree'

const DISPATCH_ICON =
  'M189 375Q189 338 207 306.5Q225 275 256.5 257Q288 239 325 239H675Q712 239 743.5 257Q775 275 793 306.5Q811 338 811 375V775Q811 812 793 843.5Q775 875 743.5 893Q712 911 675 911H325Q288 911 256.5 893Q225 875 207 843.5Q189 812 189 775ZM261 375V775Q261 802 279.5 820.5Q298 839 325 839H675Q702 839 720.5 820.5Q739 802 739 775V375Q739 348 720.5 329.5Q702 311 675 311H325Q298 311 279.5 329.5Q261 348 261 375ZM411 275H339V100Q339 85 349.5 74.5Q360 64 375 64Q390 64 400.5 74.5Q411 85 411 100ZM661 275H589V150Q589 135 599.5 124.5Q610 114 625 114Q640 114 650.5 124.5Q661 135 661 150ZM375 539H625A36 36 0 0 1 625 611H375A36 36 0 0 1 375 539Z'

type TriPanelKind = 'analytics' | 'map' | 'training' | 'calendar'

interface TriPanelShellProps {
  kind: TriPanelKind
  id?: string
  panelClass?: string
  page?: boolean
  label: string
  title: string
  barClass?: string
  titleClass?: string
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
  title,
  barClass,
  titleClass,
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
          {!page && (
            <span class={`tri-ana-title${titleClass ? ` ${titleClass}` : ''}`} data-i18n={title}>
              {title}
            </span>
          )}
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
    title="analytics"
    controls={
      <div class="tri-analytics-search-wrap tri-key-anchor tri-key-search">
        <input
          class="tri-ana-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk|strength|yoga, sort:distance|cadence|pace)"
          aria-label="search analytics"
          autocomplete="off"
        />
        <ShortcutHint>/</ShortcutHint>
        <button
          class="tri-ana-compare-toggle"
          type="button"
          aria-pressed="false"
          aria-label="compare activities"
          aria-controls="tri-analytics-results"
          data-i18n-aria-label="compare activities"
          data-site-cursor-action
        >
          <svg class="tri-ana-compare-icon" viewBox="0 0 1000 1000" aria-hidden="true">
            <path d={DISPATCH_ICON} />
          </svg>
        </button>
      </div>
    }
  >
    <div id="tri-analytics-results" class="tri-ana-results" aria-hidden="true" />
    <div id="tri-analytics-detail" class="tri-ana-detail" aria-hidden="true" />
    {ANALYTICS_CATALOG.map(definition => (
      <div class="tri-ana-block" data-chart={definition.key}>
        {page && renderData && (
          <AnalyticsServerPanel definition={definition} data={renderData.analytics} />
        )}
      </div>
    ))}
  </TriPanelShell>
)

export const MapPanel = ({ page }: { page?: boolean }) => (
  <TriPanelShell
    kind="map"
    page={page}
    label="triathlon route maps"
    title="map"
    barClass="tri-map-bar"
    titleClass="tri-map-title"
    bodyClass="tri-map-body"
    controls={
      <div class="tri-map-search-wrap tri-key-anchor tri-key-search">
        <input
          class="tri-ana-search tri-map-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk, sort:distance|pace|cadence)"
          aria-label="search routes"
          autocomplete="off"
        />
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

const TrainingTree = ({ nodes }: { nodes: readonly TrainingTreeNode[] }) => {
  const lines: JSX.Element[] = []
  const blank = ' '
  const segment = (bar: boolean): string => (bar ? `│${blank.repeat(3)}` : blank.repeat(4))
  const visit = (node: TrainingTreeNode, last: boolean, ancestors: boolean[]): void => {
    const prefix = ancestors.map(segment).join('') + (last ? '└── ' : '├── ')
    lines.push(
      <div class="tri-training-tree-line">
        <span class="tri-training-tree-prefix">{prefix}</span>
        <button class="tri-training-tree-link" type="button" data-target={node.id}>
          {node.label}
        </button>
      </div>,
    )
    node.children.forEach((child, index) =>
      visit(child, index === node.children.length - 1, [...ancestors, !last]),
    )
  }
  nodes.forEach((node, index) => visit(node, index === nodes.length - 1, []))
  return <>{lines}</>
}

const TrainingPlanDocument = ({ plans }: { plans: readonly TrainingPlan[] }) => {
  const plan = plans[0]
  if (!plan) return null
  const renderDocument = deriveTrainingDocument(plan)
  const authors = plan.author
    .split(',')
    .map(author => author.trim())
    .filter(Boolean)
    .join(', ')
  return (
    <>
      <div class="tri-training-list">
        <div class="tri-training-plans" aria-label="training plans" data-tri-ssr="true">
          {plans.map((item, index) => (
            <button
              class={`tri-ana-ritem${index === 0 ? ' tri-ana-ritem--sel' : ''}`}
              type="button"
              data-plan={index}
            >
              <span class="tri-ana-ritem-t">{item.meta}</span>
              <span class="tri-ana-ritem-s">
                {[item.distance, item.target].filter(Boolean).join(' · ')}
              </span>
            </button>
          ))}
        </div>
        <div class="tri-training-tree" aria-label="plan sections" data-tri-ssr="true">
          <TrainingTree nodes={renderDocument.tree} />
        </div>
      </div>
      <div
        class="tri-ana-detail tri-training-doc"
        aria-hidden="false"
        data-keyboard-scroll
        data-tri-ssr="true"
      >
        <div class="tri-pop-head tri-training-head">
          <span class="tri-pop-date tri-training-meta-name">{plan.meta}</span>
          <ul class="tri-training-meta">
            {plan.distance && (
              <li>
                <span class="tri-training-meta-k">distance</span>
                <span class="tri-training-meta-v">{plan.distance}</span>
              </li>
            )}
            {plan.date && (
              <li>
                <span class="tri-training-meta-k">date</span>
                <span class="tri-training-meta-v">{plan.date}</span>
              </li>
            )}
            {plan.target && (
              <li>
                <span class="tri-training-meta-k">objectif</span>
                <span class="tri-training-meta-v">{plan.target}</span>
              </li>
            )}
            {authors && (
              <li>
                <span class="tri-training-meta-k">avec</span>
                <span class="tri-training-meta-v">{authors}</span>
              </li>
            )}
          </ul>
        </div>
        <div
          class="tri-training-render"
          dangerouslySetInnerHTML={{ __html: renderDocument.html }}
        />
      </div>
    </>
  )
}

export const TrainingPanel = ({
  page,
  renderData,
}: {
  page?: boolean
  renderData?: TriathlonRenderData
}) => (
  <TriPanelShell
    kind="training"
    page={page}
    label="triathlon training plan"
    title="training"
    barClass="tri-training-bar"
    titleClass="tri-training-title"
    bodyClass="tri-training-body"
    controls={
      <div class="tri-training-search-wrap tri-key-anchor tri-key-search">
        <input
          class="tri-ana-search tri-training-search"
          type="search"
          placeholder="search plans (meta, distance, target)"
          aria-label="search training plans"
          autocomplete="off"
        />
        <ShortcutHint>/</ShortcutHint>
        <div class="tri-ana-results tri-training-results" aria-hidden="true" />
      </div>
    }
  >
    {page && renderData?.plans[0] ? (
      <TrainingPlanDocument plans={renderData.plans} />
    ) : (
      <>
        <div class="tri-training-list">
          <div class="tri-training-plans" aria-label="training plans" />
          <div class="tri-training-tree" aria-label="plan sections" />
        </div>
        <div class="tri-ana-detail tri-training-doc" aria-hidden="true" data-keyboard-scroll />
      </>
    )}
  </TriPanelShell>
)

export const CalendarOverlay = ({ renderData }: { renderData?: TriathlonRenderData }) => (
  <TriPanelShell
    kind="calendar"
    id="tri-calendar-panel"
    panelClass="tri-calendar-panel"
    label="triathlon calendar"
    title="calendar"
    controls={<CalendarSourceControls id="tri-calendar" icons />}
  >
    <CalendarPanel renderData={renderData} id="tri-calendar" panel />
  </TriPanelShell>
)
