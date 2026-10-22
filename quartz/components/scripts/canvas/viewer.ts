import type { CanvasScene } from './scene'
import { installCamera, mountCamera } from './camera'
import { drawCards, installCards } from './cards'
import { bindControls } from './commands'
import { buildControls } from './controls'
import { fetchJson, type CanvasConfig, type CanvasData, type CanvasMeta } from './data'
import { bindDrag, installDrag } from './drag'
import { drawGraph, installGraph } from './graph'
import { bindCards } from './interactions'
import { disposeCanvas, isCanvasActive, type CanvasLifecycle } from './lifecycle'
import { installPreviews, mountPreviews } from './previews'

export async function renderCanvas(container: HTMLElement, lifecycle: CanvasLifecycle) {
  const canvasUrl = container.getAttribute('data-canvas')?.trim()
  if (!canvasUrl) return

  const { signal } = lifecycle.controller

  const cfgAttr = container.getAttribute('data-cfg')
  const metaUrl = container.getAttribute('data-meta')?.trim()
  // `true` is the static preview; `live` is a bounded viewer inside a page: toolbar, zoom, drag
  // and `f` for full screen, with the wheel zooming so it does not scroll the page past it
  const embedMode = container.getAttribute('data-embed')
  const isEmbed = embedMode === 'true'
  const isLive = embedMode === 'live'

  try {
    const [canvasData, metaMap] = await Promise.all([
      fetchJson<CanvasData>(canvasUrl),
      metaUrl
        ? fetchJson<Record<string, CanvasMeta>>(metaUrl)
        : Promise.resolve<Record<string, CanvasMeta>>({}),
    ])
    if (!isCanvasActive(container, lifecycle)) return

    const cfg: CanvasConfig = cfgAttr ? JSON.parse(cfgAttr) : {}
    if (isEmbed) {
      cfg.zoom = false
    }

    if (!canvasData.nodes || canvasData.nodes.length === 0) {
      container.textContent = 'Empty canvas'
      return
    }

    drawCanvas(container, lifecycle, signal, canvasData, metaMap, cfg, isEmbed, isLive)
  } catch (error) {
    if (isCanvasActive(container, lifecycle)) {
      disposeCanvas(container, lifecycle)
      console.error('Failed to render canvas:', error)
      container.innerHTML =
        '<div class="canvas-error">Failed to load canvas data. Check the console for details.</div>'
    }
  }
}

function drawCanvas(
  container: HTMLElement,
  lifecycle: CanvasLifecycle,
  signal: AbortSignal,
  canvasData: CanvasData,
  metaMap: Record<string, CanvasMeta>,
  cfg: CanvasConfig,
  isEmbed: boolean,
  isLive: boolean,
) {
  const s = {
    container,
    lifecycle,
    signal,
    canvasData,
    metaMap,
    cfg,
    isEmbed,
    isLive,
  } as CanvasScene
  installCamera(s)
  installGraph(s)
  installCards(s)
  installPreviews(s)
  installDrag(s)
  buildControls(s)
  mountCamera(s)
  bindControls(s)
  drawGraph(s)
  drawCards(s)
  mountPreviews(s)
  bindCards(s)
  bindDrag(s)
}
