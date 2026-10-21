const MAX_SCALE = 3
const RACK_SLOPE = Math.tan(Math.PI / 6)
const GLIDE_MS = 90
const SCROLL_SETTLE_MS = 140
const ZOOM_GLIDE_MS = 90
const ZOOM_REST_MS = 200

export function createVaultCamera(
  root: HTMLElement,
  viewport: HTMLElement,
  svg: SVGSVGElement,
  dialog: HTMLDialogElement,
  reduced: MediaQueryList,
  signal: AbortSignal,
) {
  const { x: minX, y: minY, width: W, height: H } = svg.viewBox.baseVal
  const [bayLeft, bayTop, , bayBottom] = (svg.dataset.bay ?? '').split(' ').map(Number)
  const zoomIn = root.querySelector<HTMLButtonElement>('[data-camera="in"]')
  const zoomOut = root.querySelector<HTMLButtonElement>('[data-camera="out"]')
  // The drawing sits in a scaler that carries the zoom as a composited transform. Any change on
  // the SVG root itself relayouts the whole scene, so the root keeps its size.
  const canvas = document.createElement('div')
  canvas.className = 'vault-canvas'
  const scaler = document.createElement('div')
  scaler.className = 'vault-scaler'
  // While the camera moves, a blank sheet over the drawing takes the pointer, so cans sliding under
  // a still pointer are not each hit-tested and hovered. It turns on through an inline style: an
  // attribute or class change would wake the site's :has() rules and restyle the whole page.
  const shield = document.createElement('div')
  shield.className = 'vault-shield'
  svg.remove()
  scaler.append(svg)
  canvas.append(scaler, shield)
  viewport.append(canvas)

  // Cached sizes: the viewport's from the observer, the drawing's from its viewBox.
  let viewW = viewport.clientWidth
  let viewH = viewport.clientHeight
  let scale = 1
  let fitScale = 1
  let aim: { left: number; top: number } | undefined
  let glideFrame = 0
  let glideAt = 0
  let inGlideStep = false
  let writtenScroll = { left: -1, top: -1 }
  let scrollingTimer: ReturnType<typeof setTimeout> | undefined
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
  let restTimer: ReturnType<typeof setTimeout> | undefined
  const liveTransform = ([dx, dy, k]: number[]) => `translate(${dx}px, ${dy}px) scale(${k})`
  // A scroll during the zoom moves the committed canvas; the zoom's own camera moves with it.
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
    if (restTimer !== undefined) clearTimeout(restTimer)
    restTimer = undefined
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
    // The next glide starts where the running one is now.
    const t = liveAnim?.effect?.getComputedTiming().progress ?? 1
    liveFrom = liveFrom.map((v, i) => v + (liveTo[i] - v) * t)
    if (!liveAnim) liveFrom = liveTo
    liveTo = [nx - live.vl - live.ox + live.left, ny - live.vt - live.oy + live.top, scale]
    liveAnim?.cancel()
    liveAnim = scaler.animate(
      [{ transform: liveTransform(liveFrom) }, { transform: liveTransform(liveTo) }],
      { duration: reduced.matches ? 0 : ZOOM_GLIDE_MS, fill: 'forwards' },
    )
    if (restTimer !== undefined) clearTimeout(restTimer)
    restTimer = setTimeout(commitZoom, ZOOM_REST_MS)
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
    // The drawing keeps its unit size and only its wrapper scales: a new <svg> size lays out and
    // repaints all ~10k nodes, a stall of a few hundred ms at every pause in a zoom.
    scaler.style.transform = `scale(${scale})`
    writeScroll(wx * scale + nx - ax, wy * scale + ny - ay)
  }

  const stepGlide = (now = performance.now()) => {
    glideFrame = 0
    if (signal.aborted || dialog.open || !aim) return
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
  const fit = () => {
    setScale(fitScale, viewW / 2, viewH / 2)
    const [ox, oy] = pad(fitScale)
    writeScroll(ox + (W * fitScale - viewW) / 2, oy + (H * fitScale - viewH) / 2)
  }

  viewport.addEventListener(
    'scroll',
    () => {
      if (scrollingTimer === undefined) shield.style.pointerEvents = 'auto'
      else clearTimeout(scrollingTimer)
      scrollingTimer = setTimeout(() => {
        scrollingTimer = undefined
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

  // A press ends the shield at once and finds the can under it, so a click right after a scroll or a
  // camera glide still reaches the can. A click whose press dropped the shield targets the canvas.
  // A held pan or pull also raises the shield, with the grabbing cursor. An attribute on the viewport
  // for the cursor would restyle the whole page through the site's :has() rules.
  let holding = false
  const hold = (on: boolean) => {
    holding = on
    shield.style.cursor = on ? 'grabbing' : ''
    if (on || scrollingTimer === undefined) shield.style.pointerEvents = on ? 'auto' : ''
  }
  const canAt = (event: MouseEvent) => {
    let hit = event.target
    if (hit === shield || hit === canvas) {
      if (scrollingTimer !== undefined) clearTimeout(scrollingTimer)
      scrollingTimer = undefined
      shield.style.pointerEvents = ''
      hit = document.elementFromPoint(event.clientX, event.clientY)
    }
    return hit instanceof Element ? hit.closest<SVGGElement>('.cn-can') : null
  }
  // A pinch delivers many wheel events a frame; they fold into one scale step per frame, about
  // the anchor of the last one. The viewport rect is read once per gesture, not per event.
  let zoomFrame = 0
  let zoomTimer: ReturnType<typeof setTimeout> | undefined
  let zoomFactor = 1
  let zoomAnchor: [number, number] = [0, 0]
  let zoomRect: DOMRect | undefined
  let zoomRectAt = 0
  const cancelZoom = () => {
    if (zoomFrame) cancelAnimationFrame(zoomFrame)
    zoomFrame = 0
    if (zoomTimer !== undefined) clearTimeout(zoomTimer)
    zoomTimer = undefined
  }
  const flushZoom = () => {
    cancelZoom()
    const factor = zoomFactor
    zoomFactor = 1
    if (factor === 1 || signal.aborted) return
    zoomLive(scale * factor, zoomAnchor[0], zoomAnchor[1])
  }
  // The next frame flushes; a throttled page (no frames) flushes on a short timer instead.
  const scheduleZoom = () => {
    if (zoomFrame || zoomTimer !== undefined) return
    zoomFrame = requestAnimationFrame(flushZoom)
    zoomTimer = setTimeout(flushZoom, 32)
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
    commitZoom()
    measure()
    if (scale < fitScale) setScale(fitScale, viewW / 2, viewH / 2)
    else layoutCanvas(scale)
  })
  measure()
  // The first frame fits the first bay's shelves to the stage height, its left post at the left
  // edge; the other bays run on to the right.
  setScale(bayBottom > bayTop ? (viewH - 48) / (bayBottom - bayTop) : fitScale, 0, 0)
  const [initOx, initOy] = pad(scale)
  writeScroll(
    initOx + (bayLeft - minX) * scale - 24,
    initOy + ((bayTop + bayBottom) / 2 - minY) * scale - viewH / 2,
  )
  observer.observe(viewport)

  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      cancelGlide()
      if (scrollingTimer !== undefined) clearTimeout(scrollingTimer)
      scrollingTimer = undefined
      cancelZoom()
      if (restTimer !== undefined) clearTimeout(restTimer)
      restTimer = undefined
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
    commitZoom,
    cancelGlide,
    setScale,
    lookAt,
    fit,
    writeScroll,
    hold,
    canAt,
    finishGlide() {
      if (!aim) return
      const target = aim
      cancelGlide()
      writeScroll(target.left, target.top)
    },
  }
}

export type VaultCamera = ReturnType<typeof createVaultCamera>
