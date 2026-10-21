import { createVaultCamera } from './base-vault/camera'
import { createVaultCardMedia } from './base-vault/card-media'
import { bindVaultGestures } from './base-vault/gestures'
import { createVaultMotion } from './base-vault/motion-layer'
import { prepareVaultScene } from './base-vault/prepare-scene'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const ZOOM_STEP = 1.25
const PULL_MS = 220
const MAX_SETTLING = 3
// A can left out this long with nothing happening on the stage slides back into the rack.
const IDLE_MS = 2400
// A released drag keeps going for this long at its last speed before the snap decides its side.
const FLICK_MS = 140
const WHEEL_COOLDOWN_MS = 220
const WHEEL_STEP_PX = 64
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'
/** Match the release velocity (progress/ms) without overshooting the remaining distance. */
function snapEase(speed: number, distance: number, duration: number): string {
  const x1 = 0.25
  const y1 = distance > 0 ? Math.min(1, Math.max(0, (speed * duration * x1) / distance)) : 0
  return `cubic-bezier(${x1}, ${Math.round(y1 * 1000) / 1000}, 0.3, 1)`
}

export function initializeVault(root: HTMLElement): void {
  if (root.dataset.vaultReady !== undefined) return
  const viewport = root.querySelector<HTMLElement>('.vault-viewport')
  const indexPane = root.querySelector<HTMLElement>('.vault-index')
  const dialog = root.querySelector<HTMLDialogElement>('.vault-card')
  if (!viewport || !indexPane || !dialog) return
  const scene = prepareVaultScene(root)
  if (!scene) return
  const { svg } = scene
  root.dataset.vaultReady = 'loading'
  document.body.dataset.vaultPage = ''
  const signal = rootNavSignal(root)
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
  const cans = new Map<number, SVGGElement>()
  for (const can of svg.querySelectorAll<SVGGElement>(':scope > .cn-can'))
    cans.set(Number(can.dataset.index), can)
  const rows = new Map<number, HTMLLIElement>()
  for (const row of indexPane.querySelectorAll<HTMLLIElement>('.vault-entry'))
    rows.set(Number(row.dataset.index), row)
  const total = rows.size
  const readout = root.querySelector<HTMLElement>('[data-vault-readout]')
  const sectionLabel = root.querySelector<HTMLElement>('[data-vault-section]')
  if (readout && !readout.textContent) readout.textContent = `– / ${total}`
  const camera = createVaultCamera(root, viewport, svg, dialog, reduced, signal)
  const motion = createVaultMotion(svg)
  let current = -1
  let cardIndex = -1
  let returnFocus: HTMLElement | undefined
  let activePullAnim: Animation | undefined
  let pullFrom = 0
  let idleTimer: ReturnType<typeof setTimeout> | undefined
  const pulled = new Map<SVGGElement, SVGGElement>()
  const progress = new WeakMap<SVGGElement, number>()
  const settling = new Map<SVGGElement, { finish: () => void }>()
  let wheelAt = 0
  let wheelDelta = 0

  const titleOf = (row?: HTMLLIElement) =>
    row?.querySelector('.vault-entry-title')?.textContent ?? ''

  const cancelIdle = () => {
    if (idleTimer !== undefined) clearTimeout(idleTimer)
    idleTimer = undefined
  }
  const armIdle = () => {
    cancelIdle()
    if (current < 0 || dialog.open) return
    idleTimer = setTimeout(() => {
      idleTimer = undefined
      if (!signal.aborted && !dialog.open && !gestures.active && current >= 0) release()
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

  const pullProgress = (clone: SVGGElement) => {
    const held = progress.get(clone) ?? 0
    if (clone.dataset.pulling !== undefined || !activePullAnim) return held
    const eased = activePullAnim.effect?.getComputedTiming().progress ?? 0
    return pullFrom + (1 - pullFrom) * eased
  }

  const sendHome = (can: SVGGElement, clone: SVGGElement, from: number, speed = 0) => {
    pulled.delete(can)
    delete clone.dataset.out
    delete clone.dataset.pulling
    progress.delete(clone)
    clone.style.transform = ''
    // A copy on its way home must not take the pointer from the can in its slot.
    clone.style.pointerEvents = 'none'
    if (reduced.matches || from < 0.08 || typeof clone.animate !== 'function') {
      motion.remove(clone)
      if (!pulled.has(can)) can.style.opacity = ''
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
      motion.remove(clone)
      if (!pulled.has(can)) can.style.opacity = ''
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
    const start = from ?? pullProgress(clone)
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
    scene.ensure(index)
    const clone = motion.mount(can, titleOf(rows.get(index)))
    if (clone) pulled.set(can, clone)
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
    indexPane.scrollBy({ top: delta, behavior: 'instant' })
  }

  const mark = (index: number, revealRow: boolean) => {
    const row = rows.get(index)
    if (!row) return
    if (revealRow) scrollRowIntoView(row)
    current = index
    row.dataset.current = ''
    row.querySelector('.vault-entry-select')?.setAttribute('aria-current', 'true')
    if (readout) readout.textContent = `${index + 1} / ${total}`
    if (sectionLabel)
      sectionLabel.textContent =
        row.closest('details')?.querySelector('.vault-index-label')?.textContent ?? ''
  }

  const select = (index: number, moveCamera = true, revealRow = true) => {
    const can = cans.get(index)
    const row = rows.get(index)
    if (!can || !row) return
    const [px, py] = pullOf(can)
    if (index === current) {
      if (revealRow) scrollRowIntoView(row)
      if (moveCamera)
        camera.lookAt(Number(can.dataset.cx) + px * 0.6, Number(can.dataset.cy) + py * 0.6)
      return
    }
    if (revealRow) scrollRowIntoView(row)
    release()
    const clone = mountClone(index)
    if (!clone) return
    clone.dataset.out = ''
    progress.set(clone, 1)
    activePullAnim?.cancel()
    activePullAnim = undefined
    pullFrom = 0
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
    if (moveCamera) {
      camera.lookAt(Number(can.dataset.cx) + px * 0.6, Number(can.dataset.cy) + py * 0.6)
    }
  }

  const card = createVaultCardMedia(dialog, rows, reduced)
  const renderCard = (enter = false) => card.render(cardIndex >= 0 ? cardIndex : current, enter)

  const openCard = (index: number) => {
    const wasOpen = dialog.open
    if (!wasOpen) {
      select(index, true, true)
      camera.finishGlide()
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
    if (clone.dataset.pulling === undefined) clone.dataset.pulling = ''
    progress.set(clone, t)
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
    progress.set(clone, 1)
    if (reduced.matches || t >= 0.999 || typeof clone.animate !== 'function') {
      restOut(clone)
      return
    }
    const duration = Math.max(90, Math.round(PULL_MS * (1 - t)))
    pullFrom = t
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

  const gestures = bindVaultGestures({
    viewport,
    dialog,
    signal,
    camera,
    cans,
    pulled,
    pullOf,
    pullProgress,
    current: () => current,
    release,
    mountClone,
    placePull,
    finishPull,
    openCard,
    cancelPull: () => {
      activePullAnim?.cancel()
      activePullAnim = undefined
    },
  })
  const { clearDragged } = gestures
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
      else if (key === '+' || key === '=') camera.setScale(camera.scale * ZOOM_STEP)
      else if (key === '-') camera.setScale(camera.scale / ZOOM_STEP)
      else if (key === '0') camera.fit()
      else return
      event.preventDefault()
    },
    { signal },
  )
  const cleanupCard = () => {
    card.clear()
    if (signal.aborted) {
      cardIndex = -1
      returnFocus = undefined
      return
    }
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
      const actionCamera = event.target.closest<HTMLElement>('[data-camera]')?.dataset.camera
      if (actionCamera === 'fit') camera.fit()
      else if (actionCamera)
        camera.setScale(actionCamera === 'in' ? camera.scale * ZOOM_STEP : camera.scale / ZOOM_STEP)
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
      disposeEscape()
      clearDragged()
      cancelIdle()
      activePullAnim?.cancel()
      activePullAnim = undefined
      for (const s of settling.values()) s.finish()
      settling.clear()
      card.clear()
      if (dialog.open) closeCard()
      for (const [can, clone] of pulled) {
        motion.remove(clone)
        can.style.opacity = ''
      }
      pulled.clear()
      motion.dispose()
      scene.dispose()
      delete root.dataset.vaultReady
      if (!document.querySelector('.vault[data-vault-ready]'))
        delete document.body.dataset.vaultPage
    },
    { once: true },
  )
  scene.start(signal)
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
