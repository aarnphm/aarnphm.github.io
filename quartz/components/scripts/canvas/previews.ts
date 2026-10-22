import { zoomIdentity } from 'd3'
import type { NodeData } from './data'
import type { CanvasScene } from './scene'
import { mountHtmlCanvasPreview } from '../html-canvas-preview'
import { cleanupHydratedRoot } from '../root-lifecycle'
import { hydrateFileNodeContent } from './content'
import { addCanvasCleanup, isCanvasActive } from './lifecycle'

export function installPreviews(s: CanvasScene) {
  s.schedulePreviewWork = function schedulePreviewWork() {
    if (s.previewTimer || s.previewIdle || s.previewJobs.length === 0 || s.signal.aborted) return
    s.previewTimer = window.setTimeout(
      () => {
        s.previewTimer = 0
        s.previewIdle = window.requestIdleCallback(() => {
          s.previewIdle = 0
          if (s.mousePanning || s.nodeDragging || performance.now() - s.lastCameraInput < 120) {
            s.schedulePreviewWork()
            return
          }
          let job = s.previewJobs.shift()
          while (job && !job.isActive()) {
            job.run(false)
            job = s.previewJobs.shift()
          }
          job?.run(true)
          s.schedulePreviewWork()
        })
      },
      Math.max(0, 120 - (performance.now() - s.lastCameraInput)),
    )
  }

  s.waitForPreviewWork = function waitForPreviewWork(isActive: () => boolean): Promise<boolean> {
    if (s.signal.aborted) return Promise.resolve(false)
    return new Promise(resolve => {
      s.previewJobs.push({ run: resolve, isActive })
      s.schedulePreviewWork()
    })
  }

  s.updatePreviewVisibility = function updatePreviewVisibility() {
    const transform = s.currentTransform
    const margin = 160
    for (const preview of s.previews) {
      const d = preview.node
      const left = transform.x + d.x * transform.k
      const top = transform.y + d.y * transform.k
      const visible =
        transform.k >= 0.5 &&
        left < s.width + margin &&
        top < s.height + margin &&
        left + d.width * transform.k > -margin &&
        top + d.height * transform.k > -margin
      preview.renderer?.update(transform.k, visible)
      if (preview.visible === visible) continue
      const wasVisible = preview.visible
      preview.visible = visible
      const generation = ++preview.generation
      const content = preview.content
      if (!content) {
        preview.element.style.visibility = visible ? '' : 'hidden'
        continue
      }
      // SVG already clips offscreen cards. Inherited visibility would restyle every child
      // while panning; retire the HTML during idle work instead.
      if (visible) {
        if (content.children.length > 0 && content.dataset.loading === 'false') continue
        void hydrateFileNodeContent(
          content,
          d,
          () =>
            isCanvasActive(s.container, s.lifecycle) &&
            preview.visible === true &&
            preview.generation === generation,
          s.waitForPreviewWork,
        ).then(() => {
          if (preview.visible && preview.generation === generation)
            content.scrollTop = preview.scrollTop
        })
      } else if (wasVisible) {
        preview.scrollTop = content.scrollTop
        const isActive = () =>
          isCanvasActive(s.container, s.lifecycle) &&
          preview.visible === false &&
          preview.generation === generation
        void s.waitForPreviewWork(isActive).then(active => {
          if (!active || !isActive()) return
          cleanupHydratedRoot(content)
          content.textContent = d.description ?? 'Loading preview...'
          content.dataset.loading = 'true'
        })
      }
    }
  }

  s.focusCard = function focusCard(host: SVGGElement, d: NodeData, zoomToCard: boolean) {
    if (s.focusedNode && s.focusedNode !== host) s.focusedNode.classList.remove('is-focused')
    s.focusedNode = host
    host.classList.add('is-focused')
    if (zoomToCard && s.zoomBehavior) {
      s.svg.interrupt()
      s.isFitted = false
      const fit = s.getBoundsTransform({
        minX: d.x,
        minY: d.y,
        maxX: d.x + d.width,
        maxY: d.y + d.height,
      })
      const scale = Math.max(0.65, Math.min(1, fit.k))
      const transform = zoomIdentity
        .translate(
          s.width / 2 - scale * (d.x + d.width / 2),
          s.height / 2 - scale * (d.y + d.height / 2),
        )
        .scale(scale)
      s.svg
        .transition()
        .duration(s.reducedMotion.matches ? 0 : 180)
        .call(s.zoomBehavior.transform, transform)
    }
  }
}

export function mountPreviews(s: CanvasScene) {
  s.previews = []

  s.node.each(function (d) {
    const element = this.querySelector<SVGForeignObjectElement>('.canvas-card-preview')
    if (!element) return
    const content = element.querySelector<HTMLElement>('.node-content')
    const renderer = content ? mountHtmlCanvasPreview(content, d.width, d.height, s.signal) : null
    if (renderer) addCanvasCleanup(s.lifecycle, renderer.dispose)
    s.previews.push({
      node: d,
      element,
      content: this.querySelector<HTMLElement>('.node-file-content'),
      visible: null,
      scrollTop: 0,
      generation: 0,
      renderer,
    })
  })

  addCanvasCleanup(s.lifecycle, () => {
    for (const preview of s.previews) {
      if (preview.content) cleanupHydratedRoot(preview.content)
    }
  })

  s.node
    .append('rect')
    .attr('class', 'node-border-overlay')
    .attr('width', d => d.width)
    .attr('height', d => d.height)
    .attr('stroke', 'var(--gray)')
    .attr('stroke-width', 1.5)
    .attr('fill', 'none')
    .attr('pointer-events', 'none')
}
