import type { CanvasScene } from './scene'

export function bindControls(s: CanvasScene) {
  if (!s.isEmbed) {
    s.container.addEventListener(
      'wheel',
      event => {
        if (s.isHelpOpen || !s.zoomBehavior) return
        const target = event.target
        if (
          !event.metaKey &&
          !event.ctrlKey &&
          s.focusedNode &&
          target instanceof Element &&
          s.focusedNode.contains(target) &&
          target.closest('.canvas-card-preview')
        ) {
          // Native HTML-in-Canvas can retarget wheel events to the bitmap surface.
          // Its HTML scroller still owns the card's reading position.
          if (target.closest('.canvas-html-surface') && !target.closest('.node-content')) {
            const content = s.focusedNode.querySelector<HTMLElement>(
              '.node-file-content, .node-text',
            )
            if (content) {
              const unit =
                event.deltaMode === WheelEvent.DOM_DELTA_LINE
                  ? 16
                  : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
                    ? content.clientHeight
                    : 1
              event.preventDefault()
              event.stopPropagation()
              content.scrollBy(event.deltaX * unit, event.deltaY * unit)
            }
          }
          return
        }

        event.preventDefault()
        event.stopPropagation()
        s.svg.interrupt()
        s.isFitted = false
        const unit =
          event.deltaMode === WheelEvent.DOM_DELTA_LINE
            ? 16
            : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
              ? s.height
              : 1
        const dx = event.deltaX * unit
        const dy = event.deltaY * unit
        if (event.metaKey || event.ctrlKey || (s.isLive && !event.shiftKey)) {
          const rect = s.container.getBoundingClientRect()
          const scale = s.currentScale * Math.pow(2, -dy * 0.002 * (event.ctrlKey ? 10 : 1))
          s.svg.call(s.zoomBehavior.scaleTo, scale, [
            event.clientX - rect.left - s.container.clientLeft,
            event.clientY - rect.top - s.container.clientTop,
          ])
        } else {
          const panX = event.shiftKey ? dx || dy : dx
          const panY = event.shiftKey ? 0 : dy
          // D3 translations use world units; wheel deltas use screen pixels.
          s.svg.call(s.zoomBehavior.translateBy, -panX / s.currentScale, -panY / s.currentScale)
        }
      },
      { passive: false, signal: s.signal },
    )
  }

  s.showHelp = () => {
    if (s.helpModal) {
      s.helpModal.classList.add('is-visible')
      s.helpModal.showModal()
      s.isHelpOpen = true
    }
  }

  s.hideHelp = () => {
    if (s.helpModal) {
      s.helpModal.classList.remove('is-visible')
      s.helpModal.close()
      s.isHelpOpen = false
    }
  }

  if (s.helpModal) {
    s.helpModal.addEventListener(
      'cancel',
      event => {
        event.preventDefault()
        s.hideHelp()
      },
      { signal: s.signal },
    )
    const helpBackdrop = s.helpModal.querySelector<HTMLElement>('.canvas-help-backdrop')
    const helpClose = s.helpModal.querySelector<HTMLButtonElement>('.canvas-help-close')

    if (helpBackdrop) {
      helpBackdrop.addEventListener('click', s.hideHelp, { signal: s.signal })
    }

    if (helpClose) {
      helpClose.addEventListener('click', s.hideHelp, { signal: s.signal })
    }
  }

  if (s.isEmbed) {
    const navigateTo = s.container.getAttribute('data-navigate-to')
    if (navigateTo) {
      const overlay = document.createElement('a')
      overlay.className = 'canvas-embed-overlay'
      overlay.style.cssText =
        'position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1000; cursor: pointer;'
      overlay.setAttribute('aria-label', `view full canvas: ${navigateTo}`)
      overlay.href = navigateTo.startsWith('/') ? navigateTo : `/${navigateTo}`
      s.container.appendChild(overlay)
    }
  }

  if (s.toolbar) {
    s.toolbar.querySelectorAll<HTMLButtonElement>('[data-action]').forEach(button => {
      button.addEventListener(
        'click',
        event => {
          event.stopPropagation()
          const action = button.dataset.action

          if (action === 'help') {
            s.showHelp()
            return
          }

          if (action === 'fullscreen') {
            s.toggleFullscreen()
            return
          }
          if (!s.zoomBehavior) return
          s.svg.interrupt()
          s.isFitted = action === 'zoom-fit'

          switch (action) {
            case 'zoom-in':
              s.svg
                .transition()
                .duration(s.reducedMotion.matches ? 0 : 180)
                .call(s.zoomBehavior.scaleBy, 1.3)
              break
            case 'zoom-out':
              s.svg
                .transition()
                .duration(s.reducedMotion.matches ? 0 : 180)
                .call(s.zoomBehavior.scaleBy, 0.7)
              break
            case 'zoom-reset':
              s.svg
                .transition()
                .duration(s.reducedMotion.matches ? 0 : 180)
                .call(s.zoomBehavior.scaleTo, 1)
              break
            case 'zoom-fit':
              if (s.bounds) {
                s.applyBoundsTransform(s.bounds, true)
              }
              break
          }
        },
        { signal: s.signal },
      )
    })
  }

  s.toggleFullscreen = () => {
    if (document.fullscreenElement === s.container) void document.exitFullscreen()
    else void s.container.requestFullscreen?.()
  }

  s.handleKeydown = (event: KeyboardEvent) => {
    const target = event.target
    if (
      target instanceof Element &&
      target.closest('input, textarea, select, [contenteditable="true"]')
    )
      return
    if (event.key === 'Escape') {
      s.hideHelp()
      if (s.focusedNode) {
        s.focusedNode.classList.remove('is-focused')
        s.focusedNode = null
      }
      return
    }
    if (s.isEmbed || s.isHelpOpen || !s.zoomBehavior) return
    // a live canvas shares its page, so its keys act only while focus is inside it
    if (s.isLive && !(target instanceof Element && s.container.contains(target))) return
    if (s.isLive && event.key === 'f' && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault()
      s.toggleFullscreen()
      return
    }
    if (target instanceof Element && !s.container.contains(target) && target !== document.body)
      return
    if (event.key === ' ') {
      if (target instanceof Element && target.closest('button, a')) return
      event.preventDefault()
      s.spacePressed = true
      return
    }
    if (event.key === '0' || (event.shiftKey && event.code === 'Digit1')) {
      event.preventDefault()
      s.svg.interrupt()
      s.isFitted = true
      if (s.bounds) s.applyBoundsTransform(s.bounds, true)
      return
    }
    // On AZERTY, Shift+0 types "0" and fits above; layouts with a distinct Shift+0 reset to 100%.
    if (event.shiftKey && event.code === 'Digit0' && !event.metaKey && !event.ctrlKey) {
      event.preventDefault()
      s.svg.interrupt()
      s.isFitted = false
      s.svg
        .transition()
        .duration(s.reducedMotion.matches ? 0 : 180)
        .call(s.zoomBehavior.scaleTo, 1)
      return
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return
    if (target instanceof Element && target.closest('.node-content')) return
    const step = event.shiftKey ? 160 : 40
    const pan: Record<string, [number, number]> = {
      ArrowLeft: [step, 0],
      ArrowRight: [-step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    }
    const delta = pan[event.key]
    if (delta) {
      event.preventDefault()
      s.svg.interrupt()
      s.isFitted = false
      s.svg.call(s.zoomBehavior.translateBy, delta[0] / s.currentScale, delta[1] / s.currentScale)
    } else if (event.key === '+' || event.key === '=' || event.key === '-') {
      event.preventDefault()
      s.svg.interrupt()
      s.isFitted = false
      s.svg.call(s.zoomBehavior.scaleBy, event.key === '-' ? 1 / 1.3 : 1.3)
    }
  }

  document.addEventListener('keydown', s.handleKeydown, { signal: s.signal })

  document.addEventListener(
    'keyup',
    event => {
      if (event.key === ' ') s.spacePressed = false
    },
    { signal: s.signal },
  )

  window.addEventListener(
    'blur',
    () => {
      s.spacePressed = false
    },
    { signal: s.signal },
  )
}
