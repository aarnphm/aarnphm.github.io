import { JSX } from 'preact'
import type { TriathlonMaintenance } from '../../util/triathlon-maintenance'
import type { TriathlonRenderData } from '../triathlon/render-data'
import { QuartzComponent, QuartzComponentProps } from '../../types/component'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
import { triathlonDateTree, triathlonFeedScopeFromSlug } from '../../util/triathlon-date-route'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'
import { TRI_ANALYTICS_BOOT_SCRIPT } from '../triathlon/analytics/boot'
import {
  AnalyticsPanel,
  CalcPanel,
  FeedPanel,
  MapPanel,
  OnTreePanel,
  ToolsPanel,
  TrainingPanel,
  TriathlonSubnav,
  type TriView,
} from './triathlon-panels'

interface TriathlonSubPanelProps {
  page?: boolean
  defaultDistance?: unknown
  maintenance?: TriathlonMaintenance | null
  renderData?: TriathlonRenderData
}

const PANEL: Record<
  Exclude<TriView, 'on' | 'feed'>,
  (props: TriathlonSubPanelProps) => JSX.Element
> = {
  tools: ToolsPanel,
  calc: CalcPanel,
  analytics: AnalyticsPanel,
  maps: MapPanel,
  training: TrainingPanel,
}

export const TriathlonSubPage = (view: TriView, defaultDistance?: unknown): QuartzComponent => {
  const Page: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const root = pathToRoot(fileData.slug!)
    const feedScope = view === 'on' ? triathlonFeedScopeFromSlug(fileData.slug!) : null
    return (
      <div
        class={classNames(displayClass, 'triathlon', 'tri-subpage', 'all-col', 'popover-hint')}
        data-tri-view={view}
        data-detail-path={joinSegments(root, 'static/strava-detail.json')}
        data-analytics-path={joinSegments(root, 'static/analytics.json')}
        data-oura-detail-path={joinSegments(root, 'static/oura-detail.json')}
        data-training-path={joinSegments(root, 'static/training.json')}
        tabindex={-1}
        data-keyboard-scroll-scope
      >
        <TriathlonSubnav active={view} root={root} />
        {view === 'on' ? (
          <OnTreePanel
            root={root}
            tree={triathlonDateTree(
              fileData.stravaPayload?.days
                ? Object.fromEntries(fileData.stravaPayload.days.map(day => [day.date, day]))
                : (fileData.stravaPayload?.details ?? {}),
              feedScope?.prefix,
            )}
          />
        ) : view === 'feed' ? (
          <FeedPanel />
        ) : (
          PANEL[view]({
            page: true,
            defaultDistance,
            maintenance: fileData.triathlonMaintenance,
            renderData: fileData.triathlonRenderData,
          })
        )}
      </div>
    )
  }
  Page.css = style
  Page.beforeDOMLoaded = TRI_ANALYTICS_BOOT_SCRIPT
  Page.afterDOMLoaded = script
  return Page
}
