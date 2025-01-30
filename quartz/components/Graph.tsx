import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/graph.inline"
import style from "./styles/graph.scss"
import { classNames } from "../util/lang"

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
  focusOnHover?: boolean
}

export const defaultOptions: Partial<D3Config> | undefined = {
  drag: true,
  zoom: true,
  depth: -1,
  scale: 0.8,
  repelForce: 0.9,
  centerForce: 0.4,
  linkDistance: 60,
  fontSize: 0.5,
  opacityScale: 1,
  showTags: true,
  removeTags: [],
  focusOnHover: true,
}

export default ((opts?: Partial<D3Config>) => {
  const cfg = JSON.stringify({ ...defaultOptions, ...opts })
  const Graph: QuartzComponent = ({ displayClass }: QuartzComponentProps) => (
    <div class={classNames(displayClass, "graph")}>
      <div id="global-graph-outer">
        <div id="global-graph-container" data-cfg={cfg} />
      </div>
    </div>
  )

  Graph.css = style
  Graph.afterDOMLoaded = script

  return Graph
}) satisfies QuartzComponentConstructor
