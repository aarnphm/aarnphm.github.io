import type { CameraAction } from '../panel'
import type { Point } from '../scene-model'
import type { ViewHost } from './view-host'
import { cameraTarget, toPx } from '../stage'

/** Books either side of the focus whose shelves mount ahead of the camera. */
const LOOKAHEAD = 12

/** One zoom step, as the garden's buttons. */
export const ZOOM = 1.35
/** A released pan closes about 95% of the way back in three of these. */
const GLIDE_MS = 90
/** A scroll this soon after a wheel or touch move is the user's, and selects. */
const USER_SCROLL_MS = 250

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

export function onCamera(this: ViewHost, action: CameraAction): void {
  if (action === 'in') this.zoomBy(ZOOM)
  else if (action === 'out') this.zoomBy(1 / ZOOM)
  else if (action === 'fullscreen') {
    if (this.stage.fullscreen) void document.exitFullscreen().catch(() => undefined)
    else void this.root.requestFullscreen().catch(() => undefined)
  } else {
    this.zoom = 1
    this.pan = [0, 0]
    this.releasing = false
    this.requestPlace()
  }
}

export function maxZoom(this: ViewHost): number {
  const { scale } = this.frame
  return Math.max(2, scale * 4) / scale
}

export function zoomBy(this: ViewHost, factor: number): void {
  const zoom = clamp(this.zoom * factor, 1, this.maxZoom())
  if (zoom === this.zoom) return
  const ratio = zoom / this.zoom
  this.zoom = zoom
  this.pan = [this.pan[0] * ratio, this.pan[1] * ratio]
  this.requestPlace()
}

export function setControls(this: ViewHost): void {
  this.panel.setCamera({
    canIn: this.zoom < this.maxZoom() - 0.001,
    canOut: this.zoom > 1.001,
    fullscreen: this.stage.fullscreen,
  })
}

export function onFullscreen(this: ViewHost): void {
  const full = this.stage.fullscreen
  if (full === this.fullscreen) return
  this.fullscreen = full
  this.root.toggleClass('is-fullscreen', full)
  if (full) this.observer?.observe(this.root)
  else this.observer?.unobserve(this.root)
  this.stage.reset()
  this.restoreFocus = true
  this.setControls()
  this.schedule()
}

export function place(this: ViewHost): void {
  const [ax, ay] = this.stage.anchor()
  const z = this.zoom
  const x = ax + z * (this.base[0] - ax)
  const y = ay + z * (this.base[1] - ay)
  if (z !== 1 || this.pan[0] !== 0 || this.pan[1] !== 0) {
    const { stageWidth, sceneHeight } = this.stage.metrics
    const w = this.size[0] * z
    const h = this.size[1] * z
    const ix = Math.min(stageWidth, w) * 0.3
    const iy = Math.min(sceneHeight, h) * 0.3
    this.pan = [
      clamp(x + this.pan[0], ix - w, stageWidth - ix) - x,
      clamp(y + this.pan[1], iy - h, sceneHeight - iy) - y,
    ]
  }
  const offset: Point = [x + this.pan[0], y + this.pan[1]]
  this.offset = offset
  const move = `translate3d(${offset[0]}px, ${offset[1]}px, 0)`
  this.world.style.transform = z === 1 ? move : `${move} scale(${z})`
  this.root.toggleClass('is-zoomed', z !== 1)
}

export function band(this: ViewHost): [number, number] {
  return this.stage.band(this.offset, this.zoom)
}

export function requestPlace(this: ViewHost): void {
  this.placeRaf ??= requestAnimationFrame(now => {
    this.placeRaf = null
    if (this.releasing) {
      const k = 1 - Math.exp(-Math.min(64, Math.max(0, now - this.releaseAt)) / GLIDE_MS)
      this.releaseAt = now
      this.pan = [this.pan[0] * (1 - k), this.pan[1] * (1 - k)]
      if (Math.hypot(this.pan[0], this.pan[1]) < 0.5) {
        this.pan = [0, 0]
        this.releasing = false
      }
    }
    this.place()
    this.layers.check(this.band(), this.stage.span(this.offset, this.zoom))
    this.setControls()
    this.requestSettle()
    if (this.releasing) this.requestPlace()
  })
}

export function releasePan(this: ViewHost): void {
  if (this.drag.dragging || this.releasing || (this.pan[0] === 0 && this.pan[1] === 0)) return
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    this.pan = [0, 0]
    return
  }
  this.releasing = true
  this.releaseAt = performance.now()
  this.requestPlace()
}

export function centre(this: ViewHost, index: number): Point {
  return (this.centres[index] ??= toPx(this.frame, this.scene!.books[index].centre))
}

export function reach(this: ViewHost, index: number): Point {
  let reach = this.reaches[index]
  if (reach === undefined) {
    const book = this.scene!.books[index]
    let left = Infinity
    let right = -Infinity
    for (const [x] of book.faces.edge) {
      left = Math.min(left, x)
      right = Math.max(right, x)
    }
    const px = (x: number) => (x + book.pull[0] - this.frame.minX) * this.frame.scale
    reach = [px(left), px(right)]
    this.reaches[index] = reach
  }
  return reach
}

export function onScroll(this: ViewHost, evt: Event): void {
  if (evt.currentTarget !== this.stage.scroller || this.scrollRaf !== null) return
  this.lastScroll = performance.now()
  const user = this.lastScroll - this.userScroll < USER_SCROLL_MS
  this.scrollRaf = requestAnimationFrame(() => {
    this.scrollRaf = null
    this.camera(false, undefined, user)
  })
}

export function camera(this: ViewHost, sync = false, scrollTop?: number, user = false): void {
  const scene = this.scene
  const settings = this.settings
  if (!scene || !settings) return
  const count = scene.books.length
  const t = this.stage.progress(count, settings.scrollStep, scrollTop)
  // An empty case sits in the middle of the stage; otherwise the focused spine holds the anchor.
  const target = count > 0 ? cameraTarget(i => this.centre(i), count, t) : this.emptyTarget()
  let offset: Point
  if (count > 0) {
    const a = this.reach(Math.floor(t))
    const b = Math.ceil(t) < count ? this.reach(Math.ceil(t)) : a
    const k = t - Math.floor(t)
    const lift: Point = [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]
    offset = this.stage.offset(target, lift, this.reserve)
  } else {
    const { stageWidth, sceneHeight } = this.stage.metrics
    offset = [stageWidth / 2 - target[0], sceneHeight / 2 - target[1]]
  }
  this.base = offset
  const focus = count > 0 ? Math.round(t) : -1
  if (focus !== this.focus) this.releasePan()
  this.place()
  if (focus !== this.focus) this.setFocus(focus)
  if (sync) this.syncSelection()
  // A user's scroll selects the book at the reading position, and slides it out again.
  else if (user && focus >= 0 && (focus !== this.selected || !this.lifted)) this.select(focus, true)
  // A rebuild mounts only the shelves on the stage; the idle pass mounts the rest.
  const urgent = this.layers.check(
    this.band(),
    sync ? this.stage.span(this.offset, this.zoom) : undefined,
  )
  // One shelf mount per focus change: the ones the camera needs now, else one ahead of it.
  // Shelf markup is rendered in idle time, so a mount here only parses it.
  if (focus !== this.focusMounts) {
    this.focusMounts = focus
    if (urgent === 0 && !sync)
      this.layers.mountNext(this.ahead(), this.band(), this.stage.metrics.stageHeight)
  }
  this.requestSettle()
}

export function ahead(this: ViewHost): number[] {
  const books = this.scene?.books ?? []
  const order: number[] = []
  for (let d = 1; d <= LOOKAHEAD; d++)
    for (const i of [this.focus + this.heading * d, this.focus - this.heading * d]) {
      const layer = books[i] ? this.shelfLayer.get(books[i].shelf) : undefined
      if (layer !== undefined) order.push(layer)
    }
  return order
}

export function emptyTarget(this: ViewHost): Point {
  const [minX, minY, maxX, maxY] = this.scene?.bounds ?? [0, 0, 0, 0]
  return toPx(this.frame, [(minX + maxX) / 2, (minY + maxY) / 2])
}

export function setFocus(this: ViewHost, focus: number): void {
  if (this.focus >= 0 && focus >= 0) this.heading = focus < this.focus ? -1 : 1
  this.focus = focus
  const book = this.scene?.books[focus]
  if (book) this.focusPath = book.path
  this.rove(focus)
  if (!book) {
    this.preload.clear()
    return
  }
  this.preloadAround()
  this.plugin.covers.focus(this, focus)
}
