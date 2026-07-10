import { JSX } from 'preact'
import { QuartzComponent, QuartzComponentProps } from '../../types/component'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
import { triathlonDateTree, triathlonFeedScopeFromSlug } from '../../util/triathlon-date-route'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'
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

const PANEL: Record<Exclude<TriView, 'on' | 'feed'>, (props: { page?: boolean }) => JSX.Element> = {
  tools: ToolsPanel,
  calc: CalcPanel,
  analytics: AnalyticsPanel,
  maps: MapPanel,
  training: TrainingPanel,
}

export const TriathlonSubPage = (view: TriView): QuartzComponent => {
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
      >
        <TriathlonSubnav active={view} root={root} />
        {view === 'on' ? (
          <OnTreePanel
            root={root}
            title={feedScope?.title === 'feed' ? 'on' : feedScope?.title}
            tree={triathlonDateTree(fileData.stravaPayload?.details ?? {}, feedScope?.prefix)}
          />
        ) : view === 'feed' ? (
          <FeedPanel />
        ) : (
          PANEL[view]({ page: true })
        )}
      </div>
    )
  }
  Page.css = style
  Page.afterDOMLoaded = script
  return Page
}
