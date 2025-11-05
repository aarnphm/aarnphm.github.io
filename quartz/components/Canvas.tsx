import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/canvas.inline"
import style from "./styles/canvas.scss"
import { classNames } from "../util/lang"

export interface CanvasConfig {
  /**
   * Enable dragging nodes
   */
  drag: boolean

  /**
   * Enable zoom and pan
   */
  zoom: boolean

  /**
   * Force simulation strength (0-1)
   */
  forceStrength: number

  /**
   * Link distance for force simulation
   */
  linkDistance: number

  /**
   * Collision radius for nodes
   */
  collisionRadius: number

  /**
   * Use manual positioning from .canvas file
   */
  useManualPositions: boolean

  /**
   * Show file content inline in nodes
   */
  showInlineContent: boolean

  /**
   * Show preview on hover
   */
  showPreviewOnHover: boolean

  /**
   * Maximum preview length (characters)
   */
  previewMaxLength: number
}

export const defaultOptions: CanvasConfig = {
  drag: true,
  zoom: true,
  forceStrength: 0.3,
  linkDistance: 150,
  collisionRadius: 50,
  useManualPositions: true,
  showInlineContent: true,
  showPreviewOnHover: true,
  previewMaxLength: 300,
}

export default ((userOpts?: Partial<CanvasConfig>) => {
  const opts = { ...defaultOptions, ...userOpts }
  const cfg = JSON.stringify(opts)

  const Canvas: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    const canvasData = fileData.frontmatter?.canvas

    if (!canvasData) return <></>

    const dataValue = typeof canvasData === "string" ? canvasData : JSON.stringify(canvasData)
    const metaValue =
      typeof fileData.frontmatter?.canvasMeta === "string"
        ? fileData.frontmatter.canvasMeta
        : undefined

    const containerProps: Record<string, string> = {
      "data-canvas": dataValue,
      "data-cfg": cfg,
    }

    if (metaValue) {
      containerProps["data-meta"] = metaValue
    }

    return (
      <section class="canvas-component" data-canvas-title={fileData.frontmatter?.title}>
        <div class="canvas-container" {...containerProps} />
      </section>
    )
  }

  Canvas.css = style
  Canvas.afterDOMLoaded = script

  return Canvas
}) satisfies QuartzComponentConstructor<CanvasConfig>

/**
 * Standalone canvas viewer component for embedded canvases
 */
export const CanvasEmbed: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <div class={classNames(displayClass, "canvas-embed")}>
      <div class="canvas-embed-inner" />
    </div>
  )
}
