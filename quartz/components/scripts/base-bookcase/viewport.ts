import type { BookcaseScene } from './scene'

const WHEEL_SETTLE_MS = 500

export function bindViewport(s: BookcaseScene) {
  s.hoverPoint = { x: -1, y: -1 }

  s.viewport.addEventListener(
    'scroll',
    () => {
      const left = s.viewport.scrollLeft
      const top = s.viewport.scrollTop
      // Native paging or autoscroll interrupts a glide; our own camera writes retain it.
      if (left !== s.paintedScroll.left || top !== s.paintedScroll.top) {
        s.fitted = false
        s.aim = undefined
        s.cameraDirty = false
      }
      s.camera.x = s.scrollOrigin.x - left
      s.camera.y = s.scrollOrigin.y - top
      s.paintedScroll = { left, top }
      s.schedule()
    },
    { passive: true, signal: s.signal },
  )

  s.viewport.addEventListener(
    'pointerdown',
    event => {
      if (!event.isPrimary || event.button !== 0) return
      s.cancelWheel()
      const target = event.target
      const book = target instanceof Element ? target.closest<SVGGElement>('.bc-book') : null
      s.drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        camera: { ...s.camera },
        book: book ? Number(book.dataset.bookIndex) : -1,
        moved: false,
      }
      s.viewport.setPointerCapture(event.pointerId)
    },
    { signal: s.signal },
  )

  s.viewport.addEventListener(
    'pointermove',
    event => {
      if (!s.drag) {
        if (
          event.pointerType === 'touch' ||
          event.buttons !== 0 ||
          s.dialog.open ||
          s.wheelTimer !== undefined ||
          (event.clientX === s.hoverPoint.x && event.clientY === s.hoverPoint.y)
        )
          return
        s.hoverPoint = { x: event.clientX, y: event.clientY }
        const target = event.target
        const book = target instanceof Element ? target.closest<SVGGElement>('.bc-book') : null
        if (!book) return
        const index = Number(book.dataset.bookIndex)
        if (index >= 0 && index !== s.focus) s.select(index, true)
        return
      }
      if (event.pointerId !== s.drag.id) return
      const dx = event.clientX - s.drag.x
      const dy = event.clientY - s.drag.y
      if (!s.drag.moved && Math.hypot(dx, dy) < 5) return
      s.drag.moved = true
      s.fitted = false
      s.viewport.dataset.dragging = ''
      s.aim = undefined
      s.camera = { ...s.drag.camera, x: s.drag.camera.x + dx, y: s.drag.camera.y + dy }
      s.cameraDirty = true
      s.schedule()
    },
    { signal: s.signal },
  )

  s.viewport.addEventListener(
    'pointerup',
    event => {
      if (!s.drag || event.pointerId !== s.drag.id) return
      const gesture = s.drag
      s.drag = undefined
      delete s.viewport.dataset.dragging
      if (s.viewport.hasPointerCapture(event.pointerId))
        s.viewport.releasePointerCapture(event.pointerId)
      if (!gesture.moved && gesture.book >= 0) s.openPreview(gesture.book)
      else if (!gesture.moved && s.focus >= 0) s.deselect()
      // A settled camera may start the detail decodes the drag held back.
      else if (gesture.moved) s.schedule()
    },
    { signal: s.signal },
  )

  s.cancelDrag = () => {
    s.drag = undefined
    delete s.viewport.dataset.dragging
  }

  s.viewport.addEventListener('pointercancel', s.cancelDrag, { signal: s.signal })

  s.viewport.addEventListener('lostpointercapture', s.cancelDrag, { signal: s.signal })

  s.viewport.addEventListener(
    'keydown',
    event => {
      if (event.target !== s.viewport) return
      s.cancelWheel()
      if (event.key === 'Home') s.fit()
      else if (event.key === '+' || event.key === '=') s.zoom(1.35)
      else if (event.key === '-') s.zoom(1 / 1.35)
      else if (event.key === 'Enter' && s.focus >= 0) s.openPreview(s.focus)
      else if (event.key === 'Escape' && s.focus >= 0) s.deselect()
      else if (event.key.startsWith('Arrow')) {
        s.cameraDirty = true
        s.aim = undefined
        if (event.key === 'ArrowLeft') s.camera.x += 48
        if (event.key === 'ArrowRight') s.camera.x -= 48
        if (event.key === 'ArrowUp') s.camera.y += 48
        if (event.key === 'ArrowDown') s.camera.y -= 48
        s.fitted = false
        s.schedule()
      } else return
      event.preventDefault()
    },
    { signal: s.signal },
  )

  s.viewport.addEventListener(
    'wheel',
    event => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault()
        const bounds = s.viewport.getBoundingClientRect()
        s.zoom(
          Math.exp(-event.deltaY * 0.01),
          event.clientX - bounds.left,
          event.clientY - bounds.top,
        )
      } else {
        // Native scrolling keeps wheel momentum on the compositor. Selection waits for rest.
        s.aim = undefined
        s.cameraDirty = false
        s.camera.scale = s.painted.scale
        s.fitted = false
        s.cancelWheel()
        s.wheelTimer = window.setTimeout(s.finishWheel, WHEEL_SETTLE_MS)
      }
    },
    { passive: false, signal: s.signal },
  )

  s.root.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const cameraControl = event.target.closest<HTMLElement>('[data-camera]')
      if (cameraControl?.dataset.camera === 'fit') s.fit()
      else if (cameraControl) s.zoom(cameraControl.dataset.camera === 'in' ? 1.35 : 1 / 1.35)
      const action = event.target.closest<HTMLElement>('[data-preview]')?.dataset.preview
      if (action === 'close') s.dialog.close()
      else if (action === 'previous') s.browsePreview(-1)
      else if (action === 'next') s.browsePreview(1)
    },
    { signal: s.signal },
  )

  // Leaving lowers the book, unless the keyboard is still inside the bookcase.

  // Leaving lowers the book, unless the keyboard is still inside the bookcase.
  s.root.addEventListener(
    'pointerleave',
    () => {
      if (s.focus < 0 || s.dialog.open || s.drag) return
      const active = document.activeElement
      if (active && s.root.contains(active) && active.matches(':focus-visible')) return
      s.deselect()
    },
    { signal: s.signal },
  )

  s.root.addEventListener(
    'focusout',
    event => {
      const next = event.relatedTarget
      if (s.focus < 0 || s.dialog.open || (next instanceof Node && s.root.contains(next))) return
      if (!s.root.matches(':hover')) s.deselect()
    },
    { signal: s.signal },
  )
}
