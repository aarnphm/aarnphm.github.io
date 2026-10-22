import type { BaseType, Selection, ZoomTransform } from 'd3'
import type { ZoomBehavior } from 'd3-zoom'
import type { HtmlCanvasPreview } from '../html-canvas-preview'
import type {
  CanvasBounds,
  CanvasConfig,
  CanvasData,
  CanvasEdge,
  CanvasMeta,
  CanvasNode,
  LinkData,
  NodeData,
} from './data'
import type { SvgTextMetrics } from './labels'
import type { CanvasLifecycle } from './lifecycle'

// Every binding of one rendered canvas; the phases below share it.
export interface CanvasScene {
  container: HTMLElement
  lifecycle: CanvasLifecycle
  signal: AbortSignal
  canvasData: CanvasData
  metaMap: Record<string, CanvasMeta>
  cfg: CanvasConfig
  isEmbed: boolean
  isLive: boolean
  width: number
  height: number
  toolbar: HTMLElement | null
  helpModal: HTMLDialogElement | null
  svg: Selection<SVGSVGElement, unknown, null, undefined>
  defs: Selection<SVGDefsElement, unknown, null, undefined>
  pattern: Selection<SVGPatternElement, unknown, null, undefined>
  g: Selection<SVGGElement, unknown, null, undefined>
  groupNodeGroup: Selection<SVGGElement, unknown, null, undefined>
  edgeGroup: Selection<SVGGElement, unknown, null, undefined>
  edgeLabelGroup: Selection<SVGGElement, unknown, null, undefined>
  nodeGroup: Selection<SVGGElement, unknown, null, undefined>
  focusedNode: SVGGElement | null
  isHelpOpen: boolean
  bounds: CanvasBounds | null
  zoomBehavior: ZoomBehavior<SVGSVGElement, unknown> | null
  currentScale: number
  currentTransform: ZoomTransform
  renderFrame: number
  spacePressed: boolean
  isFitted: boolean
  mousePanning: boolean
  nodeDragging: boolean
  lastCameraInput: number
  previewTimer: number
  previewIdle: number
  previewJobs: { run: (active: boolean) => void; isActive: () => boolean }[]
  movedNodes: Set<CanvasNode>
  renderedTransform: string
  reducedMotion: MediaQueryList
  queueRender: () => void
  getBoundsTransform: (b: CanvasBounds) => { k: number; tx: number; ty: number }
  applyBoundsTransform: (b: CanvasBounds, animate: boolean) => ZoomTransform
  showHelp: () => void
  hideHelp: () => void
  toggleFullscreen: () => void
  handleKeydown: (event: KeyboardEvent) => void
  nodes: NodeData[]
  nodeMap: Map<string, CanvasNode>
  groupNodes: CanvasNode[]
  regularNodes: CanvasNode[]
  links: LinkData[]
  groupNode: Selection<BaseType | SVGGElement, CanvasNode, SVGGElement, unknown>
  groupLabels: { element: SVGTextElement; group: NodeData; text: string; fontSize: number }[]
  groupLabelSize: number
  measureGroupLabels: () => void
  arrowId: string
  edge: Selection<SVGGElement, LinkData, SVGGElement, unknown>
  labeledLinks: (CanvasEdge & { source: NodeData; target: NodeData } & { label: string })[]
  edgeLabels: Selection<
    SVGGElement,
    CanvasEdge & { source: NodeData; target: NodeData } & { label: string },
    SVGGElement,
    unknown
  >
  node: Selection<SVGGElement, CanvasNode, SVGGElement, unknown>
  fileNodes: Selection<SVGGElement, CanvasNode, SVGGElement, unknown>
  centerTitleMetrics: Map<SVGTextElement, SvgTextMetrics>
  centerTitleSize: number
  fitCenterTitles: () => void
  textNodes: Selection<SVGGElement, CanvasNode, SVGGElement, unknown>
  previews: {
    node: NodeData
    element: SVGForeignObjectElement
    content: HTMLElement | null
    visible: boolean | null
    scrollTop: number
    generation: number
    renderer: HtmlCanvasPreview | null
  }[]
  schedulePreviewWork: () => void
  waitForPreviewWork: (isActive: () => boolean) => Promise<boolean>
  updatePreviewVisibility: () => void
  focusCard: (host: SVGGElement, d: NodeData, zoomToCard: boolean) => void
  membersOf: (group: NodeData) => NodeData[]
  moveCard: (d: NodeData, dx: number, dy: number, members: NodeData[]) => void
  refreshBounds: () => void
  updatePositions: (moved?: Set<NodeData>) => void
  measureEdgeLabels: () => void
  boundsAttr: string | null
  resizeObserver: ResizeObserver
}
