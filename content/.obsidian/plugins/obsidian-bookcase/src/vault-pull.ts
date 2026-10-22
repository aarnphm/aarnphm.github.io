import type { VaultMotion } from './vault-motion'
import type { VaultScene } from './vault-scene'

const PULL_MS = 220
const MAX_SETTLING = 3
const FLICK_MS = 140
const IDLE_MS = 2400
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'

/** Match the release velocity (progress/ms) without overshooting the remaining distance. */
function snapEase(speed: number, distance: number, duration: number): string {
  const x1 = 0.25
  const y1 = distance > 0 ? Math.min(1, Math.max(0, (speed * duration * x1) / distance)) : 0
  return `cubic-bezier(${x1}, ${Math.round(y1 * 1000) / 1000}, 0.3, 1)`
}

export function pullOf(can: SVGGElement): [number, number] {
  const { style } = can
  return [
    parseFloat(style.getPropertyValue('--px')) || 0,
    parseFloat(style.getPropertyValue('--py')) || 0,
  ]
}

export interface VaultPullCallbacks {
  cans: () => Map<number, SVGGElement> | undefined
  motion: () => VaultMotion | undefined
  scene: () => VaultScene | undefined
  titleOf: (index: number) => string
  holding: () => boolean
  current: () => number
  onPutBack: (index: number) => void
  onSnapOut: (index: number) => void
  onSnapIn: (index: number, from: number, speed: number) => void
}

export function createVaultPull(reduced: MediaQueryList, callbacks: VaultPullCallbacks) {
  const pulled = new Map<SVGGElement, SVGGElement>()
  const progress = new WeakMap<SVGGElement, number>()
  const settling = new Map<SVGGElement, { finish: () => void }>()
  let activePullAnim: Animation | null = null
  let pullFrom = 0
  let idleTimer: number | null = null

  const cancelPull = () => {
    activePullAnim?.cancel()
    activePullAnim = null
  }

  const cancelIdle = () => {
    if (idleTimer !== null) window.clearTimeout(idleTimer)
    idleTimer = null
  }

  const armIdle = () => {
    cancelIdle()
    if (callbacks.current() < 0) return
    idleTimer = window.setTimeout(() => {
      idleTimer = null
      if (callbacks.current() >= 0 && !callbacks.holding()) putBack()
    }, IDLE_MS)
  }

  const restOut = (clone: SVGGElement, anim?: Animation) => {
    const land = () => {
      if (clone.dataset.out === undefined || clone.dataset.pulling !== undefined) return
      armIdle()
    }
    if (anim) anim.addEventListener('finish', land, { once: true })
    else land()
  }

  const pullProgress = (clone: SVGGElement): number => {
    const held = progress.get(clone) ?? 0
    if (clone.dataset.pulling !== undefined || !activePullAnim) return held
    const eased = activePullAnim.effect?.getComputedTiming().progress ?? 0
    return pullFrom + (1 - pullFrom) * eased
  }

  const sendHome = (can: SVGGElement, clone: SVGGElement, from: number, speed = 0) => {
    const motion = callbacks.motion()
    pulled.delete(can)
    delete clone.dataset.out
    delete clone.dataset.pulling
    progress.delete(clone)
    clone.style.transform = ''
    clone.style.pointerEvents = 'none'
    if (reduced.matches || from < 0.08 || typeof clone.animate !== 'function') {
      motion?.remove(clone)
      if (!pulled.has(can)) can.style.opacity = ''
      return
    }
    if (settling.size >= MAX_SETTLING) settling.values().next().value?.finish()
    const [px, py] = pullOf(can)
    const duration = Math.max(90, Math.round(PULL_MS * from))
    let done = false
    let timer: number | undefined
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
      if (timer !== undefined) window.clearTimeout(timer)
      anim.onfinish = null
      anim.cancel()
      settling.delete(can)
      motion?.remove(clone)
      if (!pulled.has(can)) can.style.opacity = ''
    }
    anim.onfinish = finish
    timer = window.setTimeout(finish, duration + 20)
    settling.set(can, { finish })
  }

  const putBack = (from?: number, speed = 0) => {
    cancelIdle()
    const cur = callbacks.current()
    const cans = callbacks.cans()
    if (!cans || cur < 0) return
    const can = cans.get(cur)
    callbacks.onPutBack(cur)
    const clone = can && pulled.get(can)
    if (!can || !clone) {
      cancelPull()
      return
    }
    const start = from ?? pullProgress(clone)
    cancelPull()
    sendHome(can, clone, start, speed)
  }

  const mountClone = (index: number): SVGGElement | undefined => {
    const can = callbacks.cans()?.get(index)
    const motion = callbacks.motion()
    const title = callbacks.titleOf(index)
    if (!can || !motion || !title) return undefined
    const existing = pulled.get(can)
    if (existing) return existing
    settling.get(can)?.finish()
    callbacks.scene()?.ensure(index)
    const clone = motion.mount(can, title)
    if (clone) pulled.set(can, clone)
    return clone
  }

  const pullOut = (index: number, can: SVGGElement) => {
    const [px, py] = pullOf(can)
    const clone = mountClone(index)
    if (!clone) return
    clone.dataset.out = ''
    progress.set(clone, 1)
    cancelPull()
    pullFrom = 0
    if (!reduced.matches && typeof clone.animate === 'function') {
      const anim = clone.animate(
        [{ transform: 'translate(0px, 0px)' }, { transform: `translate(${px}px, ${py}px)` }],
        { duration: PULL_MS, easing: EASE },
      )
      activePullAnim = anim
      anim.onfinish = () => {
        if (activePullAnim === anim) activePullAnim = null
      }
      restOut(clone, anim)
    } else restOut(clone)
  }

  const placePull = (index: number, t: number) => {
    const can = callbacks.cans()?.get(index)
    const clone = can && pulled.get(can)
    if (!can || !clone) return
    const [px, py] = pullOf(can)
    cancelPull()
    cancelIdle()
    if (clone.dataset.pulling === undefined) clone.dataset.pulling = ''
    progress.set(clone, t)
    clone.style.transform = `translate(${px * t}px, ${py * t}px)`
  }

  const finishPull = (index: number, t: number, speed = 0) => {
    const can = callbacks.cans()?.get(index)
    const clone = can && pulled.get(can)
    if (!can || !clone) return
    const [px, py] = pullOf(can)
    const out = t + speed * FLICK_MS >= 0.5
    cancelPull()
    delete clone.dataset.pulling
    clone.style.transform = ''
    if (!out) {
      if (callbacks.current() === index) callbacks.onSnapIn(index, t, -speed)
      else sendHome(can, clone, t, -speed)
      return
    }
    clone.dataset.out = ''
    progress.set(clone, 1)
    if (callbacks.current() !== index) callbacks.onSnapOut(index)
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
      if (activePullAnim === anim) activePullAnim = null
    }
    restOut(clone, anim)
  }

  const clear = (motion?: VaultMotion) => {
    cancelPull()
    cancelIdle()
    for (const s of settling.values()) s.finish()
    settling.clear()
    for (const [can, clone] of pulled) {
      motion?.remove(clone)
      can.style.opacity = ''
    }
    pulled.clear()
  }

  return {
    pulled,
    has: (can: SVGGElement) => pulled.has(can),
    cancelPull,
    cancelIdle,
    putBack,
    mountClone,
    pullOut,
    placePull,
    finishPull,
    pullProgress,
    clear,
  }
}
