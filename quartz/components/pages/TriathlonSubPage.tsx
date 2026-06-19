import { JSX } from 'preact'
import { QuartzComponent, QuartzComponentProps } from '../../types/component'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'
import {
  AnalyticsPanel,
  MapPanel,
  ToolsPanel,
  TrainingPanel,
  TriathlonSubnav,
  type TriView,
} from './triathlon-panels'

const PANEL: Record<TriView, () => JSX.Element> = {
  tools: ToolsPanel,
  analytics: AnalyticsPanel,
  maps: MapPanel,
  training: TrainingPanel,
}

export const TriathlonSubPage = (view: TriView): QuartzComponent => {
  const Panel = PANEL[view]
  const Page: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const root = pathToRoot(fileData.slug!)
    return (
      <div
        class={classNames(displayClass, 'triathlon', 'tri-subpage', 'all-col', 'popover-hint')}
        data-tri-view={view}
        data-detail-path={joinSegments(root, 'static/strava-detail.json')}
        data-analytics-path={joinSegments(root, 'static/analytics.json')}
        data-training-path={joinSegments(root, 'static/training.json')}
      >
        <TriathlonSubnav active={view} root={root} />
        <Panel />
      </div>
    )
  }
  Page.css = style
  Page.afterDOMLoaded = script
  return Page
}
