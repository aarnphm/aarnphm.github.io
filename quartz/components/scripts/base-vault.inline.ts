import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const SVG_NS = 'http://www.w3.org/2000/svg'
const ZOOM_STEP = 1.25
const MAX_SCALE = 3
// 1000 ft rim titles are 9.5 units tall in the drawing; this scale shows them at about 9 px.
const READ_SCALE = 0.95
const DRAG_PX = 4
// A press on a can locks as a pull once it has moved this far, so a short slip still clicks.
const PULL_ARM_PX = 8
// Within this angle of the pull axis the drag slides the can; a wider drag pans the rack.
const PULL_ALIGN = Math.cos((41 * Math.PI) / 180)
const PULL_MS = 220
// Hover pulls a can after the pointer rests this long, so sweeping across the rack does not leave
// a trail of cans in transit.
const HOVER_MS = 90
const BAKE_MS = 120
const UPGRADE_DELAY_MS = 300
const PRELOAD = 3
const MAX_SETTLING = 1
const WHEEL_COOLDOWN_MS = 220
const WHEEL_STEP_PX = 64
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'

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

/** Embossed rings and the round lid label, laid on the pulled can's lid. */
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
  group.append(
    node('circle', { class: 'cn-rings', r: String(r) }),
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

  // Cached sizes: the viewport's from the observer, the drawing's from its viewBox.
  let viewW = viewport.clientWidth
  let viewH = viewport.clientHeight
  let scale = 1
  let bakedScale = 1
  let fitScale = 1
  let bakeTimer: number | undefined
  const behavior = (): ScrollBehavior => (reduced.matches ? 'instant' : 'smooth')
  const offset = (k = scale) => [Math.max(0, (viewW - W * k) / 2), Math.max(0, (viewH - H * k) / 2)]

  const applySceneScale = () => {
    const ratio = bakedScale > 0 ? scale / bakedScale : 1
    if (Math.abs(ratio - 1) < 0.001) {
      svg.style.transform = ''
      svg.style.willChange = 'auto'
    } else {
      svg.style.transform = `scale(${ratio})`
      svg.style.willChange = 'transform'
    }
  }

  const bakeScale = (nextScale = scale, ax = viewW / 2, ay = viewH / 2) => {
    if (bakeTimer !== undefined) {
      window.clearTimeout(bakeTimer)
      bakeTimer = undefined
    }
    const [ox, oy] = offset(bakedScale)
    const sx = (viewport.scrollLeft + ax - ox) / scale
    const sy = (viewport.scrollTop + ay - oy) / scale
    bakedScale = nextScale
    svg.setAttribute('width', String(Math.round(W * bakedScale)))
    svg.setAttribute('height', String(Math.round(H * bakedScale)))
    applySceneScale()
    const [nx, ny] = offset(bakedScale)
    viewport.scrollTo({
      left: sx * bakedScale + nx - ax,
      top: sy * bakedScale + ny - ay,
      behavior: 'instant',
    })
  }

  const setScale = (value: number, ax = viewW / 2, ay = viewH / 2, immediate = false) => {
    scale = Math.min(MAX_SCALE, Math.max(fitScale, value))
    if (zoomOut) zoomOut.disabled = scale <= fitScale * 1.001
    if (zoomIn) zoomIn.disabled = scale >= MAX_SCALE * 0.999
    if (immediate) {
      bakeScale(scale, ax, ay)
      return
    }
    applySceneScale()
    if (bakeTimer !== undefined) window.clearTimeout(bakeTimer)
    bakeTimer = window.setTimeout(() => bakeScale(scale, ax, ay), BAKE_MS)
  }

  const lookAt = (x: number, y: number) => {
    const [ox, oy] = offset(bakedScale)
    viewport.scrollTo({
      left: (x - minX) * scale + ox - viewW / 2,
      top: (y - minY) * scale + oy - viewH / 2,
      behavior: behavior(),
    })
  }
  const measure = () => {
    viewW = viewport.clientWidth
    viewH = viewport.clientHeight
    fitScale = Math.min(1, (viewW - 32) / W, (viewH - 32) / H)
  }
  const fit = () => {
    setScale(fitScale, viewW / 2, viewH / 2, true)
    viewport.scrollTo({ left: 0, top: 0, behavior: 'instant' })
  }

  let current = -1
  let cardIndex = -1
  let returnFocus: HTMLElement | undefined
  let hover: number | undefined
  let activePullAnim: Animation | undefined
  let pulledAt = 0
  const pulled = new Map<SVGGElement, SVGGElement>()
  const settling = new Map<SVGGElement, { finish: () => void }>()
  const preloaded = new Map<string, HTMLImageElement>()
  let upgradeTimer: number | undefined
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
      window.clearTimeout(upgradeTimer)
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

  const pullOf = (can: SVGGElement): [number, number] => {
    const style = can.style
    return [
      parseFloat(style.getPropertyValue('--px')) || 0,
      parseFloat(style.getPropertyValue('--py')) || 0,
    ]
  }

  const pullProgress = (clone: SVGGElement, px: number, py: number) => {
    if (clone.dataset.pulling === undefined && activePullAnim) {
      const elapsed = performance.now() - pulledAt
      if (elapsed < 16) return 0
      if (elapsed >= PULL_MS) return 1
      const u = 1 - elapsed / PULL_MS
      return 1 - u * u * u
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

  const sceneScale = () => {
    const laid = svg.width.baseVal.value
    const rendered = svg.getBoundingClientRect().width
    return laid > 0 && rendered > 0 ? rendered / laid : 1
  }

  const sendHome = (can: SVGGElement, clone: SVGGElement, from: number) => {
    pulled.delete(can)
    delete clone.dataset.out
    delete clone.dataset.pulling
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
    const duration = Math.max(1, Math.round(PULL_MS * from))
    let done = false
    let timer: number | undefined
    const anim = clone.animate(
      [
        { transform: `translate(${px * from}px, ${py * from}px)` },
        { transform: 'translate(0px, 0px)' },
      ],
      { duration, easing: EASE, fill: 'both' },
    )
    const finish = () => {
      if (done) return
      done = true
      if (timer !== undefined) window.clearTimeout(timer)
      anim.onfinish = null
      anim.cancel()
      settling.delete(can)
      clone.remove()
      if (!pulled.has(can)) delete can.dataset.lifted
    }
    anim.onfinish = finish
    timer = window.setTimeout(finish, duration + 20)
    settling.set(can, { finish })
  }

  const release = (from?: number) => {
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
    sendHome(can, clone, start)
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
    can.dataset.lifted = ''
    pulled.set(can, clone)
    lift.append(clone)
    return clone
  }

  const scrollRowIntoView = (row: HTMLLIElement) => {
    const group = row.closest('details')
    if (group && !group.open) group.setAttribute('open', '')
    const box = row.getBoundingClientRect()
    const pane = indexPane.getBoundingClientRect()
    if (box.top >= pane.top && box.bottom <= pane.bottom) return
    const delta = box.top - pane.top - pane.height / 2 + box.height / 2
    if (typeof indexPane.scrollBy === 'function') {
      indexPane.scrollBy({ top: delta, behavior: behavior() })
    } else {
      indexPane.scrollTo({ top: indexPane.scrollTop + delta, behavior: behavior() })
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
    if (!can || !row || index === current) return
    if (revealRow) reveal(row)
    release()
    const clone = mountClone(index)
    if (!clone) return
    clone.dataset.out = ''
    const [px, py] = pullOf(can)
    activePullAnim?.cancel()
    activePullAnim = undefined
    pulledAt = performance.now()
    if (!reduced.matches && typeof clone.animate === 'function') {
      activePullAnim = clone.animate(
        [{ transform: 'translate(0px, 0px)' }, { transform: `translate(${px}px, ${py}px)` }],
        { duration: PULL_MS, easing: EASE },
      )
    }
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
    const open = dialog.querySelector<HTMLAnchorElement>('.vault-card-open')
    if (open) open.href = href
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
      upgradeTimer = window.setTimeout(() => {
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
      if (current < 0) return
      cardIndex = current
      const active = document.activeElement
      returnFocus = active instanceof HTMLElement && active !== document.body ? active : viewport
      dialog.showModal()
      renderCard(true)
    } else {
      cardIndex = index
      renderCard(false)
    }
  }
  const step = (delta: number) => {
    if (dialog.open) {
      const base = cardIndex < 0 ? current : cardIndex
      const target = base < 0 ? (delta > 0 ? 0 : total - 1) : base + delta
      if (target < 0 || target >= total) return
      cardIndex = target
      renderCard(false)
      return
    }
    const target = current < 0 ? (delta > 0 ? 0 : total - 1) : current + delta
    if (target < 0 || target >= total) return
    select(target, true, true)
  }

  const placePull = (index: number, t: number) => {
    const can = cans.get(index)
    const clone = can && pulled.get(can)
    if (!can || !clone) return
    const [px, py] = pullOf(can)
    clone.dataset.pulling = ''
    clone.style.transition = 'none'
    clone.style.transform = `translate(${px * t}px, ${py * t}px)`
  }

  const finishPull = (index: number, t: number) => {
    const can = cans.get(index)
    const clone = can && pulled.get(can)
    if (!can || !clone) return
    const [px, py] = pullOf(can)
    const out = t >= 0.5
    activePullAnim?.cancel()
    activePullAnim = undefined
    delete clone.dataset.pulling
    clone.style.transition = ''
    clone.style.transform = ''
    if (!out) {
      if (current === index) release(t)
      else sendHome(can, clone, t)
      return
    }
    if (current !== index) {
      release()
      mark(index, false)
    }
    clone.dataset.out = ''
    if (reduced.matches || t >= 0.999 || typeof clone.animate !== 'function') return
    activePullAnim = clone.animate(
      [
        { transform: `translate(${px * t}px, ${py * t}px)` },
        { transform: `translate(${px}px, ${py}px)` },
      ],
      { duration: Math.max(1, Math.round(PULL_MS * (1 - t))), easing: EASE, fill: 'both' },
    )
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
  }
  let gesture: PendingGesture | PanGesture | PullGesture | undefined
  let dragged = false
  const blocksHover = () => gesture !== undefined

  viewport.addEventListener(
    'pointerdown',
    event => {
      if (event.button !== 0 || event.pointerType === 'touch' || dialog.open) return
      cancelHover()
      const node =
        event.target instanceof Element ? event.target.closest<SVGGElement>('.cn-can') : null
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
      const dx = event.clientX - gesture.x
      const dy = event.clientY - gesture.y
      if (gesture.mode === 'pending') {
        const arm = gesture.index >= 0 ? PULL_ARM_PX : DRAG_PX
        if (Math.hypot(dx, dy) < arm) return
        const len = Math.hypot(gesture.px, gesture.py)
        const align =
          gesture.index >= 0 && len > 0
            ? Math.abs(dx * gesture.px + dy * gesture.py) / (len * Math.hypot(dx, dy))
            : 0
        if (align >= PULL_ALIGN) {
          const can = cans.get(gesture.index)
          const existing = can && pulled.get(can)
          const live = existing ? pullProgress(existing, gesture.px, gesture.py) : 0
          activePullAnim?.cancel()
          activePullAnim = undefined
          const clone = existing ?? mountClone(gesture.index)
          if (!clone || !can) {
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
            placePull(gesture.index, live)
            viewport.setPointerCapture(event.pointerId)
            viewport.dataset.pulling = ''
            gesture = {
              mode: 'pull',
              id: gesture.id,
              x: gesture.x,
              y: gesture.y,
              index: gesture.index,
              px: gesture.px,
              py: gesture.py,
              start: live,
              t: live,
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
          viewport.setPointerCapture(event.pointerId)
          viewport.dataset.dragging = ''
        }
        viewport.scrollTo({
          left: gesture.left - panDx,
          top: gesture.top - panDy,
          behavior: 'instant',
        })
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
      gesture.t = t
      placePull(gesture.index, t)
    },
    { signal },
  )
  const endGesture = (event?: PointerEvent) => {
    const active = gesture
    if (!active) return
    if (event && event.pointerId !== active.id) return
    gesture = undefined
    delete viewport.dataset.dragging
    delete viewport.dataset.pulling
    if (viewport.hasPointerCapture(active.id)) viewport.releasePointerCapture(active.id)
    if (active.mode === 'pan') {
      if (active.moved) dragged = true
      return
    }
    if (active.mode === 'pull') {
      dragged = true
      finishPull(active.index, active.t)
    }
  }
  // Hovering a can pulls it and reveals its index row without moving the camera.
  const cancelHover = () => {
    if (hover !== undefined) clearTimeout(hover)
    hover = undefined
  }
  viewport.addEventListener(
    'pointerover',
    event => {
      if (event.pointerType !== 'mouse' || blocksHover() || dialog.open) return
      const can =
        event.target instanceof Element ? event.target.closest<SVGGElement>('.cn-can') : null
      const index = Number(can?.dataset.index ?? -1)
      cancelHover()
      if (index < 0 || index === current) return
      hover = window.setTimeout(() => {
        hover = undefined
        if (!signal.aborted && !dialog.open && !blocksHover()) select(index, false)
      }, HOVER_MS)
    },
    { signal },
  )
  viewport.addEventListener('pointerleave', cancelHover, { signal })
  viewport.addEventListener('pointerup', endGesture, { signal })
  viewport.addEventListener('pointercancel', endGesture, { signal })
  viewport.addEventListener('lostpointercapture', endGesture, { signal })
  viewport.addEventListener(
    'click',
    event => {
      if (dragged) {
        dragged = false
        return
      }
      const can =
        event.target instanceof Element ? event.target.closest<SVGGElement>('.cn-can') : null
      if (can) openCard(Number(can.dataset.index))
      else release()
    },
    { signal },
  )
  viewport.addEventListener(
    'keydown',
    event => {
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
  viewport.addEventListener(
    'wheel',
    event => {
      if (!event.ctrlKey && !event.metaKey) return
      event.preventDefault()
      const rect = viewport.getBoundingClientRect()
      const factor = Math.exp(-event.deltaY * 0.0025)
      setScale(
        scale * factor,
        (event.clientX ?? rect.left + viewW / 2) - rect.left,
        (event.clientY ?? rect.top + viewH / 2) - rect.top,
      )
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
  })
  measure()
  // The first frame reads: the top-left of the first bay at a scale where rim titles are legible.
  setScale(Math.max(fitScale, READ_SCALE), viewW / 2, viewH / 2, true)
  viewport.scrollTo({
    left: (startX - minX) * scale - 24,
    top: (startY - minY) * scale - 48,
    behavior: 'instant',
  })
  observer.observe(viewport)

  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      disposeEscape()
      cancelHover()
      cancelUpgrade()
      if (bakeTimer !== undefined) window.clearTimeout(bakeTimer)
      bakeTimer = undefined
      activePullAnim?.cancel()
      activePullAnim = undefined
      for (const s of settling.values()) s.finish()
      settling.clear()
      clearPreloads()
      if (dialog.open) closeCard()
      for (const clone of pulled.values()) clone.remove()
      pulled.clear()
      delete root.dataset.vaultReady
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
