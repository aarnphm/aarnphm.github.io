import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
// @ts-ignore
import script from './scripts/graph.inline'
import style from './styles/graph.scss'

export interface D3Config {
  drag: boolean
  zoom: boolean
  depth: number
  scale: number
  repelForce: number
  centerForce: number
  linkDistance: number
  fontSize: number
  opacityScale: number
  removeTags: string[]
  showTags: boolean
  label: 'hover' | 'default' | 'off'
  focusOnHover?: boolean
  enableRadial?: boolean
}

export const defaultOptions: Partial<D3Config> | undefined = {
  drag: true,
  zoom: true,
  depth: -1,
  scale: 1,
  label: 'hover',
  repelForce: 0.5,
  centerForce: 0.3,
  linkDistance: 30,
  fontSize: 0.5,
  opacityScale: 0.8,
  showTags: false,
  removeTags: [],
  focusOnHover: true,
  enableRadial: true,
}

export default ((opts?: Partial<D3Config>) => {
  const cfg = JSON.stringify({ ...defaultOptions, ...opts })
  const Graph: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <div class={classNames(displayClass, 'graph')}>
      <div class="global-graph-outer">
        <div class="global-graph-container" data-cfg={cfg} />
      </div>
    </div>
  )

  Graph.css = style
  Graph.afterDOMLoaded = script

  return Graph
}) satisfies QuartzComponentConstructor
