const MAX_SCALE = 3
const RACK_SLOPE = Math.tan(Math.PI / 6)
const GLIDE_MS = 90
const SCROLL_SETTLE_MS = 140
const ZOOM_GLIDE_MS = 90
const ZOOM_REST_MS = 200

export interface SavedVaultCamera {
  scale: number
  left: number
  top: number
}

export function createVaultCamera(
  root: HTMLElement,
  viewport: HTMLElement,
  svg: SVGSVGElement,
  reduced: MediaQueryList,
  signal: AbortSignal,
  saved?: SavedVaultCamera,
  onResize?: () => void,
) {
  const { x: minX, y: minY, width: W, height: H } = svg.viewBox.baseVal
  const [bayLeft, bayTop, , bayBottom] = (svg.dataset.bay ?? '0 0 0 0').split(' ').map(Number)
  const zoomIn = root.querySelector<HTMLButtonElement>('[data-camera="in"]')
  const zoomOut = root.querySelector<HTMLButtonElement>('[data-camera="out"]')
  const win = root.ownerDocument.defaultView ?? window
  // The drawing sits in a scaler that carries the zoom as a composited transform. Any change on
  // the SVG root itself relayouts the whole scene, so the root keeps its size.
  const canvas = document.createElement('div')
  canvas.className = 'vault-canvas'
  const scaler = document.createElement('div')
  scaler.className = 'vault-scaler'
  // While the camera moves, a blank sheet over the drawing takes the pointer, so cans sliding under
  // a still pointer are not each hit-tested and hovered. It turns on through an inline style: an
  // attribute or class change would wake :has() rules and restyle the page.
  const shield = document.createElement('div')
  shield.className = 'vault-shield'
  svg.remove()
  scaler.append(svg)
  canvas.append(scaler, shield)
  viewport.append(canvas)

  let viewW = viewport.clientWidth
  let viewH = viewport.clientHeight
  let scale = 1
  let fitScale = 1
  let aim: { left: number; top: number } | undefined
  let glideFrame = 0
  let glideAt = 0
  let inGlideStep = false
  let writtenScroll = { left: -1, top: -1 }
  let scrollingTimer: number | null = null
  const pad = (k = scale): [number, number] => [
    Math.round(viewW - Math.min(viewW, W * k) * 0.3),
    Math.round(viewH - Math.min(viewH, H * k) * 0.3),
  ]

  const writeScroll = (left: number, top: number) => {
    const l = Math.max(0, Math.round(left))
    const t = Math.max(0, Math.round(top))
    writtenScroll = { left: l, top: t }
    viewport.scrollTo({ left: l, top: t, behavior: 'instant' })
  }

  const cancelGlide = () => {
    aim = undefined
    if (glideFrame) {
      cancelAnimationFrame(glideFrame)
      glideFrame = 0
    }
  }

  const layoutCanvas = (k = scale) => {
    const [px, py] = pad(k)
    canvas.style.width = `${Math.round(W * k + px * 2)}px`
    canvas.style.height = `${Math.round(H * k + py * 2)}px`
    scaler.style.left = `${px}px`
    scaler.style.top = `${py}px`
  }

  // A wheel zoom plays on the compositor as short animations of the scaler toward each new scale:
  // a scale written from script lays out every rim title again, about 30 ms a step. The canvas
  // size, the scroll and the scaler's own transform catch up once, when the zoom rests.
  let live:
    | { left: number; top: number; ox: number; oy: number; vl: number; vt: number }
    | undefined
  let liveAnim: Animation | undefined
  let liveFrom = [0, 0, 1]
  let liveTo = [0, 0, 1]
  let restTimer: number | null = null
  const liveTransform = ([dx, dy, k]: number[]) => `translate(${dx}px, ${dy}px) scale(${k})`
  const followScroll = (at: NonNullable<typeof live>) => {
    const left = viewport.scrollLeft
    const top = viewport.scrollTop
    at.vl += left - at.left
    at.vt += top - at.top
    at.left = left
    at.top = top
  }
  const commitZoom = () => {
    if (!live) return
    if (restTimer !== null) win.clearTimeout(restTimer)
    restTimer = null
    followScroll(live)
    const { vl, vt } = live
    live = undefined
    layoutCanvas(scale)
    scaler.style.transform = `scale(${scale})`
    liveAnim?.cancel()
    liveAnim = undefined
    writeScroll(vl, vt)
  }
  const zoomLive = (value: number, ax: number, ay: number) => {
    cancelGlide()
    if (live) {
      followScroll(live)
    } else {
      const [ox, oy] = pad(scale)
      const left = viewport.scrollLeft
      const top = viewport.scrollTop
      live = { left, top, ox, oy, vl: left, vt: top }
      liveTo = [0, 0, scale]
    }
    const prevScale = scale
    const [px, py] = pad(prevScale)
    const wx = (live.vl + ax - px) / prevScale
    const wy = (live.vt + ay - py) / prevScale
    scale = Math.min(MAX_SCALE, Math.max(fitScale, value))
    if (zoomOut) zoomOut.disabled = scale <= fitScale * 1.001
    if (zoomIn) zoomIn.disabled = scale >= MAX_SCALE * 0.999
    const [nx, ny] = pad(scale)
    const maxLeft = Math.max(0, Math.round(W * scale + nx * 2) - viewW)
    const maxTop = Math.max(0, Math.round(H * scale + ny * 2) - viewH)
    live.vl = Math.min(maxLeft, Math.max(0, Math.round(wx * scale + nx - ax)))
    live.vt = Math.min(maxTop, Math.max(0, Math.round(wy * scale + ny - ay)))
    const t = liveAnim?.effect?.getComputedTiming().progress ?? 1
    liveFrom = liveFrom.map((v, i) => v + (liveTo[i] - v) * t)
    if (!liveAnim) liveFrom = liveTo
    liveTo = [nx - live.vl - live.ox + live.left, ny - live.vt - live.oy + live.top, scale]
    liveAnim?.cancel()
    liveAnim = scaler.animate(
      [{ transform: liveTransform(liveFrom) }, { transform: liveTransform(liveTo) }],
      { duration: reduced.matches ? 0 : ZOOM_GLIDE_MS, fill: 'forwards' },
    )
    if (restTimer !== null) win.clearTimeout(restTimer)
    restTimer = win.setTimeout(commitZoom, ZOOM_REST_MS)
  }

  const setScale = (value: number, ax = viewW / 2, ay = viewH / 2) => {
    commitZoom()
    cancelGlide()
    const prevScale = scale
    const [ox, oy] = pad(prevScale)
    const wx = (viewport.scrollLeft + ax - ox) / prevScale
    const wy = (viewport.scrollTop + ay - oy) / prevScale
    scale = Math.min(MAX_SCALE, Math.max(fitScale, value))
    if (zoomOut) zoomOut.disabled = scale <= fitScale * 1.001
    if (zoomIn) zoomIn.disabled = scale >= MAX_SCALE * 0.999
    const [nx, ny] = pad(scale)
    layoutCanvas(scale)
    scaler.style.transform = `scale(${scale})`
    writeScroll(wx * scale + nx - ax, wy * scale + ny - ay)
  }

  const stepGlide = (now = performance.now()) => {
    glideFrame = 0
    if (signal.aborted || !aim) return
    const dt = now - glideAt
    glideAt = now
    const curLeft = viewport.scrollLeft
    const curTop = viewport.scrollTop
    const dx = aim.left - curLeft
    const dy = aim.top - curTop
    if (reduced.matches || dt <= 0 || Math.hypot(dx, dy) < 1) {
      const target = aim
      aim = undefined
      writeScroll(target.left, target.top)
      return
    }
    const k = 1 - Math.exp(-Math.min(64, Math.max(8, dt)) / GLIDE_MS)
    writeScroll(curLeft + dx * k, curTop + dy * k)
    if (Math.hypot(aim.left - viewport.scrollLeft, aim.top - viewport.scrollTop) < 1) {
      const target = aim
      aim = undefined
      writeScroll(target.left, target.top)
      return
    }
    inGlideStep = true
    glideFrame = requestAnimationFrame(nextNow => {
      if (inGlideStep) {
        const target = aim
        aim = undefined
        if (target) writeScroll(target.left, target.top)
        return
      }
      stepGlide(nextNow)
    })
    inGlideStep = false
  }

  const lookAt = (x: number, y: number) => {
    commitZoom()
    const [ox, oy] = pad(scale)
    const targetLeft = Math.max(0, Math.round((x - minX) * scale + ox - viewW / 2))
    const targetTop = Math.max(0, Math.round((y - minY) * scale + oy - viewH / 2))
    if (reduced.matches) {
      cancelGlide()
      writeScroll(targetLeft, targetTop)
      return
    }
    const firstStep = !aim
    aim = { left: targetLeft, top: targetTop }
    if (firstStep) {
      glideAt = performance.now()
      const curLeft = viewport.scrollLeft
      const curTop = viewport.scrollTop
      const k = 1 - Math.exp(-16 / GLIDE_MS)
      writeScroll(curLeft + (targetLeft - curLeft) * k, curTop + (targetTop - curTop) * k)
    }
    if (!glideFrame && aim) {
      inGlideStep = true
      glideFrame = requestAnimationFrame(nextNow => {
        if (inGlideStep) {
          const target = aim
          aim = undefined
          if (target) writeScroll(target.left, target.top)
          return
        }
        stepGlide(nextNow)
      })
      inGlideStep = false
    }
  }

  const measure = () => {
    viewW = viewport.clientWidth
    viewH = viewport.clientHeight
    fitScale = Math.min(1, (viewW - 32) / W, (viewH - 32) / H)
  }

  const syncSize = () => {
    commitZoom()
    measure()
    if (scale < fitScale) setScale(fitScale, viewW / 2, viewH / 2)
    else layoutCanvas(scale)
  }

  const fit = () => {
    setScale(fitScale, viewW / 2, viewH / 2)
    const [ox, oy] = pad(fitScale)
    writeScroll(ox + (W * fitScale - viewW) / 2, oy + (H * fitScale - viewH) / 2)
  }

  viewport.addEventListener(
    'scroll',
    () => {
      if (scrollingTimer === null) shield.style.pointerEvents = 'auto'
      else win.clearTimeout(scrollingTimer)
      scrollingTimer = win.setTimeout(() => {
        scrollingTimer = null
        if (!holding) shield.style.pointerEvents = ''
      }, SCROLL_SETTLE_MS)
      if (
        Math.abs(viewport.scrollLeft - writtenScroll.left) > 1 ||
        Math.abs(viewport.scrollTop - writtenScroll.top) > 1
      ) {
        cancelGlide()
      }
    },
    { passive: true, signal },
  )

  let holding = false
  const hold = (on: boolean) => {
    holding = on
    shield.style.cursor = on ? 'grabbing' : ''
    if (on || scrollingTimer === null) shield.style.pointerEvents = on ? 'auto' : ''
  }
  const canAt = (event: MouseEvent) => {
    let hit = event.target
    if (hit === shield || hit === canvas) {
      if (scrollingTimer !== null) win.clearTimeout(scrollingTimer)
      scrollingTimer = null
      shield.style.pointerEvents = ''
      hit = root.ownerDocument.elementFromPoint(event.clientX, event.clientY)
    }
    return hit instanceof Element ? hit.closest<SVGGElement>('.cn-can') : null
  }

  let zoomFrame = 0
  let zoomTimer: number | null = null
  let zoomFactor = 1
  let zoomAnchor: [number, number] = [0, 0]
  let zoomRect: DOMRect | undefined
  let zoomRectAt = 0
  const cancelZoom = () => {
    if (zoomFrame) cancelAnimationFrame(zoomFrame)
    zoomFrame = 0
    if (zoomTimer !== null) win.clearTimeout(zoomTimer)
    zoomTimer = null
  }
  const flushZoom = () => {
    cancelZoom()
    const factor = zoomFactor
    zoomFactor = 1
    if (factor === 1 || signal.aborted) return
    zoomLive(scale * factor, zoomAnchor[0], zoomAnchor[1])
  }
  const scheduleZoom = () => {
    if (zoomFrame || zoomTimer !== null) return
    zoomFrame = requestAnimationFrame(flushZoom)
    zoomTimer = win.setTimeout(flushZoom, 32)
  }
  viewport.addEventListener(
    'wheel',
    event => {
      if (!event.ctrlKey && !event.metaKey) {
        // A sideways scroll follows the rack down its slant, so the shelves stay in the frame.
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewW : 1
        const dx = (event.deltaX || (event.shiftKey ? event.deltaY : 0)) * unit
        if (!dx) return
        event.preventDefault()
        const dy = event.deltaX && !event.shiftKey ? event.deltaY * unit : 0
        viewport.scrollBy({
          left: dx,
          top: dy + dx * RACK_SLOPE,
          behavior: Math.abs(dx) >= 40 ? 'smooth' : 'instant',
        })
        return
      }
      event.preventDefault()
      if (!zoomRect || event.timeStamp - zoomRectAt > 400) {
        zoomRect = viewport.getBoundingClientRect()
        zoomRectAt = event.timeStamp
      }
      zoomFactor *= Math.exp(-event.deltaY * 0.0025)
      zoomAnchor = [
        (event.clientX ?? zoomRect.left + viewW / 2) - zoomRect.left,
        (event.clientY ?? zoomRect.top + viewH / 2) - zoomRect.top,
      ]
      scheduleZoom()
    },
    { passive: false, signal },
  )
  const observer = new ResizeObserver(() => {
    onResize?.()
    syncSize()
  })
  measure()
  if (saved) {
    setScale(Math.max(fitScale, saved.scale), viewW / 2, viewH / 2)
    writeScroll(saved.left, saved.top)
  } else {
    setScale(bayBottom > bayTop ? (viewH - 48) / (bayBottom - bayTop) : fitScale, 0, 0)
    const [initOx, initOy] = pad(scale)
    writeScroll(
      initOx + (bayLeft - minX) * scale - 24,
      initOy + ((bayTop + bayBottom) / 2 - minY) * scale - viewH / 2,
    )
  }
  observer.observe(viewport)

  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      cancelGlide()
      if (scrollingTimer !== null) win.clearTimeout(scrollingTimer)
      scrollingTimer = null
      cancelZoom()
      if (restTimer !== null) win.clearTimeout(restTimer)
      restTimer = null
      liveAnim?.cancel()
      svg.remove()
      viewport.append(svg)
      canvas.remove()
    },
    { once: true },
  )
  return {
    get scale() {
      return scale
    },
    get holding() {
      return holding
    },
    commitZoom,
    cancelGlide,
    setScale,
    lookAt,
    fit,
    syncSize,
    writeScroll,
    hold,
    canAt,
  }
}

export type VaultCamera = ReturnType<typeof createVaultCamera>
