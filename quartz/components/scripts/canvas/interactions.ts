import type { CanvasScene } from './scene'
import { resolveFileNodeUrl } from './content'
import { addCanvasCleanup } from './lifecycle'

export function bindCards(s: CanvasScene) {
  s.node.on('focusin', function (event: FocusEvent, d) {
    if (event.target !== this) {
      if (event.target instanceof Element && event.target.closest('.node-content'))
        s.focusCard(this, d, false)
      return
    }
    const left = s.currentTransform.x + d.x * s.currentScale
    const top = s.currentTransform.y + d.y * s.currentScale
    if (
      left + d.width * s.currentScale < 0 ||
      top + d.height * s.currentScale < 0 ||
      left > s.width ||
      top > s.height
    ) {
      s.focusCard(this, d, true)
    }
  })

  s.node.on(
    'pointerdown',
    function (event: PointerEvent, d) {
      if (
        event.button === 0 &&
        event.target instanceof Element &&
        event.target.closest('a, button, [role="button"], input, textarea, select, summary') &&
        event.target.closest('.node-content')
      )
        s.focusCard(this, d, false)
    },
    true,
  )

  s.node.on('keydown', function (event: KeyboardEvent, d) {
    if (event.target !== this) return
    if (event.altKey && event.key.startsWith('Arrow') && !s.isEmbed) {
      event.preventDefault()
      event.stopPropagation()
      const step = event.shiftKey ? 16 : 1
      const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0
      const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0
      s.moveCard(d, dx, dy, s.membersOf(d))
      s.refreshBounds()
    } else if (event.key === 'Enter') {
      event.preventDefault()
      s.focusCard(this, d, true)
    }
  })

  s.groupNode.on('dblclick', (event: MouseEvent, d) => {
    if (!s.zoomBehavior) return
    event.preventDefault()
    s.svg.interrupt()
    s.isFitted = false
    const label = s.groupLabels.find(label => label.group === d)
    s.applyBoundsTransform(
      { minX: d.x, minY: d.y - (label?.fontSize ?? 24), maxX: d.x + d.width, maxY: d.y + d.height },
      true,
    )
  })

  s.node.on('dblclick', function (event: MouseEvent, d) {
    if (event.target instanceof Element && event.target.closest('a')) return
    event.preventDefault()
    s.focusCard(this, d, true)
  })

  s.node.on('click', (evt, d) => {
    const clickEvent = evt as MouseEvent
    const target = clickEvent.target
    if (!(target instanceof Element)) return
    const currentNode = evt.currentTarget as SVGGElement

    if (target.closest('.canvas-node-link')) {
      if (
        clickEvent.detail > 0 &&
        !clickEvent.metaKey &&
        !clickEvent.ctrlKey &&
        !clickEvent.altKey &&
        !clickEvent.shiftKey &&
        s.focusedNode !== currentNode
      ) {
        clickEvent.preventDefault()
        clickEvent.stopPropagation()
        s.focusCard(currentNode, d, s.currentScale < 0.5)
      }
      return
    }

    if (target.closest('a:not(.canvas-popover-link)')) return
    clickEvent.stopPropagation()

    if (d.type === 'file') {
      if (clickEvent.altKey) {
        const link = currentNode.querySelector(
          '.node-content a.canvas-popover-link',
        ) as HTMLAnchorElement | null
        if (link) {
          link.dispatchEvent(
            new MouseEvent('click', { altKey: true, bubbles: true, cancelable: true }),
          )
          return
        }
      }

      s.focusCard(currentNode, d, s.currentScale < 0.5)
      if (clickEvent.metaKey || clickEvent.ctrlKey) {
        const url = resolveFileNodeUrl(d)
        if (url) window.spaNavigate(url)
      }
    } else if (d.type === 'text') {
      s.focusCard(currentNode, d, s.currentScale < 0.5)
    } else if (d.type === 'link' && d.url) {
      window.open(d.url, '_blank', 'noopener,noreferrer')
    }
  })

  s.svg.on('click', evt => {
    const target = evt.target as SVGElement
    if (
      target.tagName === 'rect' ||
      target.tagName === 'svg' ||
      target.classList.contains('canvas-background')
    ) {
      if (s.focusedNode) {
        s.focusedNode.classList.remove('is-focused')
        s.focusedNode = null
      }
    }
  })

  addCanvasCleanup(s.lifecycle, () => {
    s.node.on('click', null)
    s.node.on('keydown', null)
    s.node.on('focusin', null)
    s.node.on('pointerdown', null)
    s.node.on('dblclick', null)
    s.groupNode.on('dblclick', null)
  })
}
