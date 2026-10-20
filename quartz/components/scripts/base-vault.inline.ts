import { lidRibs } from '../../util/base/vault/layout'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const SVG_NS = 'http://www.w3.org/2000/svg'
const ZOOM_STEP = 1.25
const MAX_SCALE = 3
// 1000 ft rim titles are 9.5 units tall in the drawing; this scale shows them at about 9 px.
const READ_SCALE = 0.95
const DRAG_PX = 5
// Within this angle of the active pull direction the drag slides the can; any other angle pans.
const PULL_ALIGN = Math.cos((24 * Math.PI) / 180)
const PULL_MS = 220
const GLIDE_MS = 90
// A wheel or pinch that pauses for less than this keeps its live scale; only a rest re-renders.
const BAKE_MS = 320
const SCROLL_SETTLE_MS = 140
const UPGRADE_DELAY_MS = 300
const PRELOAD = 3
const MAX_SETTLING = 3
// A can left out this long with nothing happening on the stage slides back into the rack.
const IDLE_MS = 2400
// A released drag keeps going for this long at its last speed before the snap decides its side.
const FLICK_MS = 140
const WHEEL_COOLDOWN_MS = 220
const WHEEL_STEP_PX = 64
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'

/**
 * A firm, non-overshooting ease whose first slope matches the speed the drag let go at, so the can
 * carries on at the hand's pace and settles instead of jumping to a new speed. `speed` is progress
 * per ms toward the target, `distance` the progress still to cover.
 */
function snapEase(speed: number, distance: number, duration: number): string {
  const x1 = 0.25
  const y1 = distance > 0 ? Math.min(1, Math.max(0, (speed * duration * x1) / distance)) : 0
  return `cubic-bezier(${x1}, ${Math.round(y1 * 1000) / 1000}, 0.3, 1)`
}

/** Two lines of at most `max` characters for the round lid label. */
function lidLines(title: string, max = 12): string[] {
  const lines = ['']
  for (const word of title.toUpperCase().split(/\s+/)) {
    const line = lines[lines.length - 1]
    if (!line || line.length + word.length + 1 <= max)
      lines[lines.length - 1] = line ? `${line} ${word}` : word
    else if (lines.length < 2) lines.push(word)
    else {
      lines[1] = `${lines[1]}…`
      break
    }
  }
  return lines.map(line => (line.length > max + 1 ? `${line.slice(0, max - 1)}…` : line))
}

/** The moulded lid relief and the round lid label, laid on the pulled can's lid. */
function lidDetail(can: SVGGElement, title: string): SVGGElement | null {
  const lid = can.querySelector('.cn-lid')
  const transform = lid?.getAttribute('transform')
  const r = Number(lid?.getAttribute('r'))
  if (!transform || !Number.isFinite(r)) return null
  const node = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string>) => {
    const element = document.createElementNS(SVG_NS, tag)
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value)
    return element
  }
  const group = node('g', { class: 'cn-lid-detail', transform })
  const size = Math.round(r * 0.075 * 10) / 10
  const text = node('text', { class: 'cn-lid-text', 'font-size': String(size) })
  const lines = lidLines(title)
  lines.forEach((line, i) => {
    const span = node('tspan', {
      x: '0',
      y: String((i - (lines.length - 1) / 2) * size * 1.25 + size * 0.35),
    })
    span.textContent = line
    text.append(span)
  })
  const relief = node('g', { class: 'cn-rings' })
  relief.append(
    node('circle', { class: 'cn-lid-ring', r: String(Math.round(r * 9) / 10) }),
    node('path', { class: 'cn-lid-ribs', d: lidRibs(r) }),
  )
  group.append(
    relief,
    node('circle', { class: 'cn-lid-label', r: String(Math.round(r * 0.28)) }),
    text,
  )
  return group
}

export function initializeVault(root: HTMLElement): void {
  if (root.dataset.vaultReady !== undefined) return
  const viewport = root.querySelector<HTMLElement>('.vault-viewport')
  const svg = root.querySelector<SVGSVGElement>('svg.vault-scene')
  const lift = svg?.querySelector<SVGGElement>('.vault-lift')
  const indexPane = root.querySelector<HTMLElement>('.vault-index')
  const dialog = root.querySelector<HTMLDialogElement>('.vault-card')
  if (!viewport || !svg || !lift || !indexPane || !dialog) return
  root.dataset.vaultReady = ''
  document.body.dataset.vaultPage = ''
  const signal = rootNavSignal(root)
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
  const { x: minX, y: minY, width: W, height: H } = svg.viewBox.baseVal
  const [startX, startY] = (svg.dataset.start ?? '0 0').split(' ').map(Number)
  const cans = new Map<number, SVGGElement>()
  for (const can of svg.querySelectorAll<SVGGElement>(':scope > .cn-can'))
    cans.set(Number(can.dataset.index), can)
  const rows = new Map<number, HTMLLIElement>()
  for (const row of indexPane.querySelectorAll<HTMLLIElement>('.vault-entry'))
    rows.set(Number(row.dataset.index), row)
  const total = rows.size
  const readout = root.querySelector<HTMLElement>('[data-vault-readout]')
  const sectionLabel = root.querySelector<HTMLElement>('[data-vault-section]')
  const zoomIn = root.querySelector<HTMLButtonElement>('[data-camera="in"]')
  const zoomOut = root.querySelector<HTMLButtonElement>('[data-camera="out"]')
  const poster = dialog.querySelector<HTMLElement>('.vault-card-poster')
  const posterImage = poster?.querySelector('img')
  const previous = dialog.querySelector<HTMLButtonElement>('[data-card="previous"]')
  const next = dialog.querySelector<HTMLButtonElement>('[data-card="next"]')

  // The drawing sits in a scaler that carries the interim zoom as a composited transform. Any
  // style change on the SVG root itself relayouts the whole scene (about 11 ms for 235 cans), so
  // the root is only touched when the scale bakes after the gesture settles.
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
  let bakedScale = 1
  let fitScale = 1
  let bakeTimer: ReturnType<typeof setTimeout> | undefined
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

  // The wrapper always carries a transform on its own layer: a transform that comes and goes
  // rebuilds the layer and lays out every rim title again.
  const applySceneScale = () => {
    const ratio = bakedScale > 0 ? scale / bakedScale : 1
    scaler.style.transform = `scale(${Math.abs(ratio - 1) < 0.001 ? 1 : ratio})`
  }

  const bakeScale = (nextScale = scale) => {
    if (bakeTimer !== undefined) {
      clearTimeout(bakeTimer)
      bakeTimer = undefined
    }
    bakedScale = nextScale
    svg.setAttribute('width', String(Math.round(W * bakedScale)))
    svg.setAttribute('height', String(Math.round(H * bakedScale)))
    layoutCanvas(bakedScale)
    applySceneScale()
  }

  const setScale = (value: number, ax = viewW / 2, ay = viewH / 2, immediate = false) => {
    cancelGlide()
    const prevScale = scale
    const [ox, oy] = pad(prevScale)
    const wx = (viewport.scrollLeft + ax - ox) / prevScale
    const wy = (viewport.scrollTop + ay - oy) / prevScale
    scale = Math.min(MAX_SCALE, Math.max(fitScale, value))
    if (zoomOut) zoomOut.disabled = scale <= fitScale * 1.001
    if (zoomIn) zoomIn.disabled = scale >= MAX_SCALE * 0.999
    const [nx, ny] = pad(scale)
    if (immediate) {
      bakeScale(scale)
    } else {
      layoutCanvas(scale)
      applySceneScale()
      if (bakeTimer !== undefined) clearTimeout(bakeTimer)
      bakeTimer = setTimeout(() => bakeScale(scale), BAKE_MS)
    }
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
    setScale(fitScale, viewW / 2, viewH / 2, true)
    const [ox, oy] = pad(fitScale)
    writeScroll(ox + (W * fitScale - viewW) / 2, oy + (H * fitScale - viewH) / 2)
  }

  let current = -1
  let cardIndex = -1
  let returnFocus: HTMLElement | undefined
  let activePullAnim: Animation | undefined
  let pulledAt = 0
  let pullFrom = 0
  let pullDuration = PULL_MS
  let idleTimer: ReturnType<typeof setTimeout> | undefined
  const pulled = new Map<SVGGElement, SVGGElement>()
  const settling = new Map<SVGGElement, { finish: () => void }>()
  const preloaded = new Map<string, HTMLImageElement>()
  let upgradeTimer: ReturnType<typeof setTimeout> | undefined
  let activeOriginal: HTMLImageElement | undefined
  let pendingPosterUrl: string | undefined
  let entranceAnim: Animation | undefined
  let entranceFlying = false
  let wheelAt = 0
  let wheelDelta = 0

  const titleOf = (row?: HTMLLIElement) =>
    row?.querySelector('.vault-entry-title')?.textContent ?? ''

  const preloadCovers = (centerIndex: number) => {
    const urls = new Set<string>()
    for (let i = centerIndex - PRELOAD; i <= centerIndex + PRELOAD; i++) {
      const url = rows.get(i)?.dataset.cover
      if (url) urls.add(url)
    }
    for (const [url, img] of preloaded) {
      if (!urls.has(url)) {
        img.removeAttribute('src')
        preloaded.delete(url)
      }
    }
    for (const url of urls) {
      if (preloaded.has(url)) continue
      const img = new Image()
      img.decoding = 'async'
      img.src = url
      preloaded.set(url, img)
    }
  }

  const clearPreloads = () => {
    for (const img of preloaded.values()) img.removeAttribute('src')
    preloaded.clear()
  }

  const cancelUpgrade = () => {
    if (upgradeTimer !== undefined) {
      clearTimeout(upgradeTimer)
      upgradeTimer = undefined
    }
    pendingPosterUrl = undefined
    if (activeOriginal) {
      activeOriginal.onload = null
      activeOriginal.onerror = null
      activeOriginal.removeAttribute('src')
      activeOriginal = undefined
    }
  }

  const flushPosterUpgrade = () => {
    if (entranceFlying || !pendingPosterUrl || !posterImage || !dialog.open) return
    posterImage.src = pendingPosterUrl
    pendingPosterUrl = undefined
  }

  const cancelIdle = () => {
    if (idleTimer !== undefined) clearTimeout(idleTimer)
    idleTimer = undefined
  }
  const armIdle = () => {
    cancelIdle()
    if (current < 0 || dialog.open) return
    idleTimer = setTimeout(() => {
      idleTimer = undefined
      if (!signal.aborted && !dialog.open && !gesture && current >= 0) release()
    }, IDLE_MS)
  }
  /** Start the idle return only after the outward slide ends. */
  const restOut = (clone: SVGGElement, anim?: Animation) => {
    const land = () => {
      if (clone.dataset.out === undefined || clone.dataset.pulling !== undefined) return
      armIdle()
    }
    if (anim) anim.addEventListener('finish', land, { once: true })
    else land()
  }

  const pullOf = (can: SVGGElement): [number, number] => {
    const style = can.style
    return [
      parseFloat(style.getPropertyValue('--px')) || 0,
      parseFloat(style.getPropertyValue('--py')) || 0,
    ]
  }

  const pullProgress = (clone: SVGGElement, px: number, py: number) => {
    if (clone.dataset.pulling !== undefined && clone.dataset.progress !== undefined) {
      const p = Number(clone.dataset.progress)
      if (Number.isFinite(p)) return Math.min(1, Math.max(0, p))
    }
    if (clone.dataset.pulling === undefined && activePullAnim) {
      const elapsed = performance.now() - pulledAt
      if (elapsed < 16) return pullFrom
      if (elapsed >= pullDuration) return 1
      const u = 1 - elapsed / pullDuration
      return pullFrom + (1 - pullFrom) * (1 - u * u * u)
    }
    if (clone.dataset.progress !== undefined) {
      const p = Number(clone.dataset.progress)
      if (Number.isFinite(p)) return Math.min(1, Math.max(0, p))
    }
    const len2 = px * px + py * py
    const resting = clone.dataset.out !== undefined ? 1 : 0
    if (len2 < 1 || typeof DOMMatrix !== 'function' || typeof getComputedStyle !== 'function')
      return resting
    const raw = getComputedStyle(clone).transform
    if (!raw || raw === 'none') return resting
    const matrix = new DOMMatrix(raw)
    if (!Number.isFinite(matrix.e) || !Number.isFinite(matrix.f)) return resting
    return Math.min(1, Math.max(0, (matrix.e * px + matrix.f * py) / len2))
  }

  const sceneScale = () => scale

  const sendHome = (can: SVGGElement, clone: SVGGElement, from: number, speed = 0) => {
    pulled.delete(can)
    delete clone.dataset.out
    delete clone.dataset.pulling
    delete clone.dataset.progress
    clone.style.transition = ''
    clone.style.transform = ''
    // A copy on its way home must not take the pointer from the can in its slot.
    clone.style.pointerEvents = 'none'
    if (reduced.matches || from < 0.08 || typeof clone.animate !== 'function') {
      clone.remove()
      if (!pulled.has(can)) delete can.dataset.lifted
      return
    }
    if (settling.size >= MAX_SETTLING) settling.values().next().value?.finish()
    const [px, py] = pullOf(can)
    const duration = Math.max(90, Math.round(PULL_MS * from))
    let done = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const anim = clone.animate(
      [
        { transform: `translate(${px * from}px, ${py * from}px)` },
        { transform: 'translate(0px, 0px)' },
      ],
      { duration, easing: snapEase(speed, from, duration), fill: 'both' },
    )
    const finish = () => {
      if (done) return
      done = true
      if (timer !== undefined) clearTimeout(timer)
      anim.onfinish = null
      anim.cancel()
      settling.delete(can)
      clone.remove()
      if (!pulled.has(can)) delete can.dataset.lifted
    }
    anim.onfinish = finish
    timer = setTimeout(finish, duration + 20)
    settling.set(can, { finish })
  }

  const release = (from?: number, speed = 0) => {
    cancelIdle()
    if (current < 0) return
    const can = cans.get(current)
    const row = rows.get(current)
    delete row?.dataset.current
    row?.querySelector('.vault-entry-select')?.removeAttribute('aria-current')
    current = -1
    if (readout) readout.textContent = `– / ${total}`
    if (sectionLabel) sectionLabel.textContent = ''
    const clone = can && pulled.get(can)
    if (!can || !clone) {
      activePullAnim?.cancel()
      activePullAnim = undefined
      return
    }
    const [px, py] = pullOf(can)
    const start = from ?? pullProgress(clone, px, py)
    activePullAnim?.cancel()
    activePullAnim = undefined
    sendHome(can, clone, start, speed)
  }

  const mountClone = (index: number) => {
    const can = cans.get(index)
    if (!can) return
    const existing = pulled.get(can)
    if (existing) return existing
    settling.get(can)?.finish()
    const clone = can.cloneNode(true) as SVGGElement
    clone.querySelector('.cn-rings')?.remove()
    const detail = lidDetail(clone, titleOf(rows.get(index)))
    if (detail) clone.append(detail)
    clone.classList.add('cn-pulled')
    can.dataset.lifted = ''
    pulled.set(can, clone)
    // Keep the whole moving solid above the shelf tags and neighbouring cans until it seats.
    lift.append(clone)
    return clone
  }

  const scrollRowIntoView = (row: HTMLLIElement) => {
    const group = row.closest('details')
    if (group && !group.open) group.setAttribute('open', '')
    const box = row.getBoundingClientRect()
    const pane = indexPane.getBoundingClientRect()
    // The group heading sticks over the top of the pane; a row under it is not in view.
    const top = pane.top + (group?.querySelector('summary')?.getBoundingClientRect().height ?? 0)
    if (box.top >= top && box.bottom <= pane.bottom) return
    const delta = box.top - pane.top - pane.height / 2 + box.height / 2
    if (typeof indexPane.scrollBy === 'function') {
      indexPane.scrollBy({ top: delta, behavior: 'instant' })
    } else {
      indexPane.scrollTo({ top: indexPane.scrollTop + delta })
    }
  }

  const reveal = (row: HTMLLIElement) => {
    scrollRowIntoView(row)
  }

  const mark = (index: number, revealRow: boolean) => {
    const row = rows.get(index)
    if (!row) return
    if (revealRow) reveal(row)
    current = index
    row.dataset.current = ''
    row.querySelector('.vault-entry-select')?.setAttribute('aria-current', 'true')
    if (readout) readout.textContent = `${index + 1} / ${total}`
    if (sectionLabel)
      sectionLabel.textContent =
        row.closest('details')?.querySelector('.vault-index-label')?.textContent ?? ''
  }

  const select = (index: number, camera = true, revealRow = true) => {
    const can = cans.get(index)
    const row = rows.get(index)
    if (!can || !row) return
    const [px, py] = pullOf(can)
    if (index === current) {
      if (revealRow) reveal(row)
      if (camera) lookAt(Number(can.dataset.cx) + px * 0.6, Number(can.dataset.cy) + py * 0.6)
      return
    }
    if (revealRow) reveal(row)
    release()
    const clone = mountClone(index)
    if (!clone) return
    clone.dataset.out = ''
    clone.dataset.progress = '1'
    activePullAnim?.cancel()
    activePullAnim = undefined
    pulledAt = performance.now()
    pullFrom = 0
    pullDuration = PULL_MS
    if (!reduced.matches && typeof clone.animate === 'function') {
      const anim = clone.animate(
        [{ transform: 'translate(0px, 0px)' }, { transform: `translate(${px}px, ${py}px)` }],
        { duration: PULL_MS, easing: EASE },
      )
      activePullAnim = anim
      anim.onfinish = () => {
        if (activePullAnim === anim) activePullAnim = undefined
      }
      restOut(clone, anim)
    } else restOut(clone)
    mark(index, false)
    if (camera) {
      lookAt(Number(can.dataset.cx) + px * 0.6, Number(can.dataset.cy) + py * 0.6)
    }
  }

  const renderCard = (enter = false) => {
    const shown = cardIndex >= 0 ? cardIndex : current
    const row = rows.get(shown)
    if (!row) return
    cancelUpgrade()
    preloadCovers(shown)
    const title = titleOf(row)
    const href = row.dataset.href ?? ''
    dialog.setAttribute('aria-label', `Vault card: ${title}`)
    const heading = dialog.querySelector<HTMLAnchorElement>('.vault-card-title')
    if (heading) {
      heading.textContent = title
      heading.href = href
    }
    const source = row.querySelector('.vault-entry-detail')
    dialog
      .querySelector('.vault-card-detail')
      ?.replaceChildren(...[...(source?.childNodes ?? [])].map(child => child.cloneNode(true)))
    const position = dialog.querySelector('.vault-card-position')
    if (position) position.textContent = `${shown + 1} / ${total}`
    if (previous) previous.disabled = shown <= 0
    if (next) next.disabled = shown >= total - 1
    if (!poster || !posterImage) return
    const thumbnail = row.dataset.cover
    const full = row.dataset.fullCover
    if (thumbnail) {
      delete poster.dataset.empty
      posterImage.src = thumbnail
    } else {
      poster.dataset.empty = ''
      posterImage.removeAttribute('src')
    }
    if (enter && !reduced.matches && typeof posterImage.animate === 'function') {
      entranceAnim?.cancel()
      entranceFlying = true
      const anim = posterImage.animate([{ transform: 'rotateY(6deg)' }, { transform: 'none' }], {
        duration: PULL_MS,
        easing: EASE,
      })
      entranceAnim = anim
      anim.onfinish = () => {
        if (entranceAnim === anim) entranceAnim = undefined
        entranceFlying = false
        flushPosterUpgrade()
      }
    }
    if (thumbnail && full && full !== thumbnail) {
      upgradeTimer = setTimeout(() => {
        upgradeTimer = undefined
        if (!dialog.open || cardIndex !== shown) return
        const original = new Image()
        original.decoding = 'async'
        activeOriginal = original
        const commit = () => {
          if (!dialog.open || cardIndex !== shown || activeOriginal !== original) return
          activeOriginal = undefined
          pendingPosterUrl = full
          flushPosterUpgrade()
        }
        original.onload = commit
        original.src = full
        if (typeof original.decode === 'function') {
          original.decode().then(commit, () => {})
        }
      }, UPGRADE_DELAY_MS)
    }
  }

  const openCard = (index: number) => {
    const wasOpen = dialog.open
    if (!wasOpen) {
      select(index, true, true)
      if (aim) {
        const target = aim
        cancelGlide()
        writeScroll(target.left, target.top)
      }
      if (current < 0) return
      cardIndex = current
      const active = document.activeElement
      returnFocus = active instanceof HTMLElement && active !== document.body ? active : viewport
      cancelIdle()
      dialog.showModal()
      renderCard(true)
    } else {
      cardIndex = index
      renderCard(false)
    }
  }
  const visibleIndices = (): number[] => {
    const list: number[] = []
    for (const [idx, row] of rows) {
      if (row.style.display !== 'none') list.push(idx)
    }
    return list.length > 0 ? list : [...rows.keys()]
  }
  const step = (delta: number) => {
    const visible = visibleIndices()
    if (dialog.open) {
      const base = cardIndex < 0 ? current : cardIndex
      const pos = visible.indexOf(base)
      const nextPos = pos < 0 ? (delta > 0 ? 0 : visible.length - 1) : pos + delta
      const target = visible[nextPos]
      if (target === undefined) return
      cardIndex = target
      renderCard(false)
      return
    }
    const pos = visible.indexOf(current)
    const nextPos = pos < 0 ? (delta > 0 ? 0 : visible.length - 1) : pos + delta
    const target = visible[nextPos]
    if (target === undefined) return
    select(target, true, true)
  }

  const placePull = (index: number, t: number) => {
    const can = cans.get(index)
    const clone = can && pulled.get(can)
    if (!can || !clone) return
    const [px, py] = pullOf(can)
    activePullAnim?.cancel()
    activePullAnim = undefined
    cancelIdle()
    clone.dataset.pulling = ''
    clone.dataset.progress = String(t)
    clone.style.transition = 'none'
    clone.style.transform = `translate(${px * t}px, ${py * t}px)`
  }

  const finishPull = (index: number, t: number, speed = 0) => {
    const can = cans.get(index)
    const clone = can && pulled.get(can)
    if (!can || !clone) return
    const [px, py] = pullOf(can)
    const out = t + speed * FLICK_MS >= 0.5
    activePullAnim?.cancel()
    activePullAnim = undefined
    delete clone.dataset.pulling
    clone.style.transition = ''
    clone.style.transform = ''
    if (!out) {
      if (current === index) release(t, -speed)
      else sendHome(can, clone, t, -speed)
      return
    }
    if (current !== index) {
      release()
      mark(index, true)
    }
    clone.dataset.out = ''
    clone.dataset.progress = '1'
    if (reduced.matches || t >= 0.999 || typeof clone.animate !== 'function') {
      restOut(clone)
      return
    }
    const duration = Math.max(90, Math.round(PULL_MS * (1 - t)))
    pulledAt = performance.now()
    pullFrom = t
    pullDuration = duration
    const anim = clone.animate(
      [
        { transform: `translate(${px * t}px, ${py * t}px)` },
        { transform: `translate(${px}px, ${py}px)` },
      ],
      { duration, easing: snapEase(speed, 1 - t, duration) },
    )
    activePullAnim = anim
    anim.onfinish = () => {
      if (activePullAnim === anim) activePullAnim = undefined
    }
    restOut(clone, anim)
  }

  type PendingGesture = {
    mode: 'pending'
    id: number
    x: number
    y: number
    left: number
    top: number
    index: number
    px: number
    py: number
  }
  type PanGesture = {
    mode: 'pan'
    id: number
    x: number
    y: number
    left: number
    top: number
    moved: boolean
  }
  type PullGesture = {
    mode: 'pull'
    id: number
    x: number
    y: number
    index: number
    px: number
    py: number
    start: number
    t: number
    /** Progress per ms, smoothed over the last moves. */
    speed: number
    at: number
  }
  let gesture: PendingGesture | PanGesture | PullGesture | undefined
  let dragged = false
  let draggedResetTimer: ReturnType<typeof setTimeout> | undefined
  const clearDragged = () => {
    dragged = false
    if (draggedResetTimer !== undefined) {
      clearTimeout(draggedResetTimer)
      draggedResetTimer = undefined
    }
  }
  const markDragged = () => {
    dragged = true
    if (draggedResetTimer !== undefined) clearTimeout(draggedResetTimer)
    draggedResetTimer = setTimeout(() => {
      draggedResetTimer = undefined
      dragged = false
    }, 0)
  }
  const capturePointer = (pointerId: number) => {
    try {
      viewport.setPointerCapture?.(pointerId)
    } catch {}
  }
  const releasePointer = (pointerId: number) => {
    try {
      if (viewport.hasPointerCapture?.(pointerId)) viewport.releasePointerCapture(pointerId)
    } catch {}
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
  // A press does not focus the stage: a focus change wakes the site's :has(:focus-visible) rules and
  // restyles every element on the page. Tab still focuses it for the keys.
  viewport.addEventListener(
    'mousedown',
    event => {
      if (event.button === 0) event.preventDefault()
    },
    { signal },
  )
  viewport.addEventListener(
    'pointerdown',
    event => {
      clearDragged()
      if (event.button !== 0 || event.isPrimary === false || dialog.open) return
      cancelGlide()
      const node = canAt(event)
      const index = Number(node?.dataset.index ?? -1)
      const can = cans.get(index)
      const [px, py] = can ? pullOf(can) : [0, 0]
      gesture = {
        mode: 'pending',
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        left: viewport.scrollLeft,
        top: viewport.scrollTop,
        index: can && px * px + py * py > 1 ? index : -1,
        px,
        py,
      }
    },
    { signal },
  )
  viewport.addEventListener(
    'pointermove',
    event => {
      if (!gesture || event.pointerId !== gesture.id) return
      if (event.buttons === 0) {
        endGesture(event)
        return
      }
      const dx = event.clientX - gesture.x
      const dy = event.clientY - gesture.y
      const dist = Math.hypot(dx, dy)
      if (gesture.mode === 'pending') {
        if (dist < DRAG_PX) return
        const len = Math.hypot(gesture.px, gesture.py)
        const dot =
          gesture.index >= 0 && len > 0 ? (dx * gesture.px + dy * gesture.py) / (len * dist) : 0
        const can = gesture.index >= 0 ? cans.get(gesture.index) : undefined
        const existing = can ? pulled.get(can) : undefined
        const isOut = Boolean(existing && existing.dataset.out !== undefined)
        const live = existing ? pullProgress(existing, gesture.px, gesture.py) : 0
        const canPullOut = !isOut && dot >= PULL_ALIGN
        const canPushIn = isOut && dot <= -PULL_ALIGN
        if (can && (canPullOut || canPushIn)) {
          const startT = canPushIn ? Math.max(0.5, live) : live
          if (current >= 0 && current !== gesture.index) release()
          activePullAnim?.cancel()
          activePullAnim = undefined
          const clone = existing ?? mountClone(gesture.index)
          if (!clone) {
            gesture = {
              mode: 'pan',
              id: gesture.id,
              x: gesture.x,
              y: gesture.y,
              left: gesture.left,
              top: gesture.top,
              moved: false,
            }
          } else {
            placePull(gesture.index, startT)
            capturePointer(event.pointerId)
            hold(true)
            gesture = {
              mode: 'pull',
              id: gesture.id,
              x: gesture.x,
              y: gesture.y,
              index: gesture.index,
              px: gesture.px,
              py: gesture.py,
              start: startT,
              t: startT,
              speed: 0,
              at: event.timeStamp,
            }
          }
        } else {
          gesture = {
            mode: 'pan',
            id: gesture.id,
            x: gesture.x,
            y: gesture.y,
            left: gesture.left,
            top: gesture.top,
            moved: false,
          }
        }
      }
      if (gesture.mode === 'pan') {
        const panDx = event.clientX - gesture.x
        const panDy = event.clientY - gesture.y
        if (!gesture.moved && Math.hypot(panDx, panDy) < DRAG_PX) return
        if (!gesture.moved) {
          gesture.moved = true
          capturePointer(event.pointerId)
          hold(true)
        }
        writeScroll(gesture.left - panDx, gesture.top - panDy)
        return
      }
      if (gesture.mode !== 'pull') return
      const k = sceneScale()
      const along =
        ((event.clientX - gesture.x) / k) * gesture.px +
        ((event.clientY - gesture.y) / k) * gesture.py
      const t = Math.min(
        1,
        Math.max(0, gesture.start + along / (gesture.px * gesture.px + gesture.py * gesture.py)),
      )
      const elapsed = event.timeStamp - gesture.at
      if (elapsed > 0) gesture.speed = gesture.speed * 0.5 + ((t - gesture.t) / elapsed) * 0.5
      gesture.at = event.timeStamp
      gesture.t = t
      placePull(gesture.index, t)
    },
    { signal },
  )
  const endGesture = (event?: PointerEvent) => {
    const active = gesture
    if (!active) return
    if (event && event.pointerId !== undefined && event.pointerId !== active.id) return
    gesture = undefined
    hold(false)
    releasePointer(active.id)
    if (active.mode === 'pan') {
      if (active.moved) markDragged()
      return
    }
    if (active.mode === 'pull') {
      markDragged()
      // A hand that stopped before letting go has no speed left to carry.
      const still = (event?.timeStamp ?? performance.now()) - active.at > 80
      finishPull(active.index, active.t, still ? 0 : active.speed)
    }
  }
  viewport.addEventListener('pointerup', endGesture, { signal })
  viewport.addEventListener('pointercancel', endGesture, { signal })
  viewport.addEventListener('lostpointercapture', endGesture, { signal })
  viewport.addEventListener(
    'click',
    event => {
      if (dragged) {
        clearDragged()
        return
      }
      const can = canAt(event)
      if (can) openCard(Number(can.dataset.index))
      else release()
    },
    { signal },
  )
  viewport.addEventListener(
    'keydown',
    event => {
      clearDragged()
      if (event.altKey || event.ctrlKey || event.metaKey) return
      const key = event.key
      if (key === 'ArrowRight' || key === 'ArrowDown') step(1)
      else if (key === 'ArrowLeft' || key === 'ArrowUp') step(-1)
      else if (key === 'Enter' || key === ' ') openCard(current < 0 ? 0 : current)
      else if (key === 'Escape' && current >= 0) release()
      else if (key === '+' || key === '=') setScale(scale * ZOOM_STEP)
      else if (key === '-') setScale(scale / ZOOM_STEP)
      else if (key === '0') fit()
      else return
      event.preventDefault()
    },
    { signal },
  )
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
    setScale(scale * factor, zoomAnchor[0], zoomAnchor[1])
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
      if (!event.ctrlKey && !event.metaKey) return
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
  const cleanupCard = () => {
    cancelUpgrade()
    if (entranceAnim) {
      entranceAnim.onfinish = null
      entranceAnim.cancel()
      entranceAnim = undefined
    }
    entranceFlying = false
    posterImage?.removeAttribute('src')
    clearPreloads()
    if (cardIndex >= 0 && cardIndex !== current) {
      select(cardIndex, true, true)
    }
    cardIndex = -1
    armIdle()
    if (returnFocus) {
      const target = returnFocus.isConnected ? returnFocus : viewport
      target.focus({ preventScroll: true })
      returnFocus = undefined
    }
  }

  const closeCard = () => {
    if (dialog.open) dialog.close()
    cleanupCard()
  }

  root.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const camera = event.target.closest<HTMLElement>('[data-camera]')?.dataset.camera
      if (camera === 'fit') fit()
      else if (camera) setScale(camera === 'in' ? scale * ZOOM_STEP : scale / ZOOM_STEP)
      const action = event.target.closest<HTMLElement>('[data-card]')?.dataset.card
      if (action === 'close') closeCard()
      else if (action) step(action === 'next' ? 1 : -1)
    },
    { signal },
  )
  indexPane.addEventListener(
    'click',
    event => {
      const row =
        event.target instanceof Element ? event.target.closest<HTMLLIElement>('.vault-entry') : null
      if (row) select(Number(row.dataset.index), true, false)
    },
    { signal },
  )
  const disposeEscape = registerEscapeHandler(dialog, closeCard)
  dialog.addEventListener(
    'click',
    event => {
      if (event.target !== dialog) return
      const box = dialog.getBoundingClientRect()
      const inside =
        event.clientX >= box.left &&
        event.clientX <= box.right &&
        event.clientY >= box.top &&
        event.clientY <= box.bottom
      if (!inside) closeCard()
    },
    { signal },
  )
  dialog.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
      event.preventDefault()
      step(event.key === 'ArrowLeft' ? -1 : 1)
    },
    { signal },
  )
  dialog.addEventListener(
    'wheel',
    event => {
      if (!dialog.open || event.ctrlKey || event.metaKey) return
      const record =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('.vault-card-record')
          : null
      const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
      if (
        record &&
        record.scrollHeight > record.clientHeight + 1 &&
        ((delta > 0 && record.scrollTop + record.clientHeight < record.scrollHeight - 1) ||
          (delta < 0 && record.scrollTop > 0))
      ) {
        return
      }
      if (!delta) return
      event.preventDefault()
      const now = performance.now()
      if (now - wheelAt < WHEEL_COOLDOWN_MS) return
      if (now - wheelAt > WHEEL_COOLDOWN_MS * 2) wheelDelta = 0
      wheelDelta += delta
      if (Math.abs(wheelDelta) < WHEEL_STEP_PX) return
      const dir = wheelDelta > 0 ? 1 : -1
      wheelDelta = 0
      wheelAt = now
      step(dir)
    },
    { passive: false, signal },
  )
  dialog.addEventListener('close', cleanupCard, { signal })

  const observer = new ResizeObserver(() => {
    measure()
    if (scale < fitScale) setScale(fitScale, viewW / 2, viewH / 2, true)
    else layoutCanvas(scale)
  })
  measure()
  if (readout && !readout.textContent) readout.textContent = `– / ${total}`
  // The first frame reads: the top-left of the first bay at a scale where rim titles are legible.
  setScale(Math.max(fitScale, READ_SCALE), viewW / 2, viewH / 2, true)
  const [initOx, initOy] = pad(scale)
  writeScroll(initOx + (startX - minX) * scale - 24, initOy + (startY - minY) * scale - 48)
  observer.observe(viewport)

  const groups = [...indexPane.querySelectorAll<HTMLDetailsElement>('.vault-index-group')]
  let searchOpen: Map<HTMLDetailsElement, boolean> | undefined
  const onSearchResults = () => {
    const query = document.querySelector<HTMLInputElement>('#base-search-input')?.value.trim() ?? ''
    const searching = query.length > 0
    if (searching && !searchOpen) searchOpen = new Map(groups.map(g => [g, g.open]))
    for (const group of groups) {
      const groupRows = [...group.querySelectorAll<HTMLLIElement>('.vault-entry')]
      const shownCount = groupRows.filter(r => r.style.display !== 'none').length
      const countEl = group.querySelector<HTMLElement>('.vault-index-count')
      if (countEl) countEl.textContent = String(shownCount)
      if (searching && shownCount > 0) group.open = true
      else if (!searching && searchOpen) group.open = searchOpen.get(group) ?? true
    }
    if (!searching) searchOpen = undefined
    for (const [idx, can] of cans) {
      const row = rows.get(idx)
      if (searching && row?.style.display === 'none') can.dataset.dimmed = ''
      else delete can.dataset.dimmed
    }
    if (current >= 0 && rows.get(current)?.style.display === 'none') release()
  }
  document.addEventListener('base-search-results', onSearchResults, { signal })

  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      disposeEscape()
      clearDragged()
      cancelIdle()
      cancelGlide()
      cancelUpgrade()
      if (bakeTimer !== undefined) clearTimeout(bakeTimer)
      bakeTimer = undefined
      if (scrollingTimer !== undefined) clearTimeout(scrollingTimer)
      scrollingTimer = undefined
      cancelZoom()
      activePullAnim?.cancel()
      activePullAnim = undefined
      for (const s of settling.values()) s.finish()
      settling.clear()
      clearPreloads()
      if (dialog.open) closeCard()
      for (const clone of pulled.values()) clone.remove()
      pulled.clear()
      svg.remove()
      viewport.append(svg)
      canvas.remove()
      delete root.dataset.vaultReady
      if (!document.querySelector('.vault[data-vault-ready]'))
        delete document.body.dataset.vaultPage
    },
    { once: true },
  )
}

document.addEventListener('nav', () => {
  const initialize = () => {
    for (const root of document.querySelectorAll<HTMLElement>('.vault')) initializeVault(root)
  }
  initialize()
  document.addEventListener('base-content-changed', initialize)
  window.addCleanup(() => document.removeEventListener('base-content-changed', initialize))
})

export default ''
