import type { BookcaseScene } from './scene'
import type { Camera } from './state'
import { createHardcoverPreview } from './preview'

// Camera glide time constant, and the lift's beats: the pull out of the slot, its wait for the
// glide, and the slide home of the copy it replaces.
const GLIDE_MS = 90

export const PULL_DELAY_MS = 160

export function mountCamera(s: BookcaseScene) {
  s.readout = s.root.querySelector<HTMLElement>('[data-bookcase-readout]')

  s.shelfLabel = s.root.querySelector<HTMLElement>('[data-bookcase-shelf]')

  s.previewArt = s.dialog.querySelector<HTMLElement>('.bookcase-preview-art')

  s.hardcover = s.previewArt ? createHardcoverPreview(s.previewArt, s.signal) : undefined

  s.previousButton = s.dialog.querySelector<HTMLButtonElement>('[data-preview="previous"]')

  s.nextButton = s.dialog.querySelector<HTMLButtonElement>('[data-preview="next"]')

  s.zoomOut = s.root.querySelector<HTMLButtonElement>('[data-camera="out"]')

  s.zoomIn = s.root.querySelector<HTMLButtonElement>('[data-camera="in"]')

  s.preloaded = new Map<string, HTMLImageElement>()

  s.camera = { x: 0, y: 0, scale: 1 }

  s.fitScale = 1

  // The fit shows one bay at the stage's height; zooming out may show every bay.

  // The fit shows one bay at the stage's height; zooming out may show every bay.
  s.minScale = 1

  s.fitWidth = Number(s.scene.dataset.fit) || s.box.width

  s.fitted = true

  s.focus = -1

  // The pull of the lifted copy, and copies still sliding home with the covers they exposed.

  // The pull of the lifted copy, and copies still sliding home with the covers they exposed.

  s.settling = new Set<{ svg: SVGSVGElement; cover?: SVGImageElement; done: () => void }>()

  // A camera the view glides toward, and when it last stepped.

  // A camera the view glides toward, and when it last stepped.

  s.steppedAt = 0

  // The last painted camera and its screen velocity in px/ms, for prefetch.

  // The last painted camera and its screen velocity in px/ms, for prefetch.
  s.painted = { x: 0, y: 0, scale: 1, at: 0 }

  s.velocity = { x: 0, y: 0 }

  s.frame = 0

  s.cancelWheel = () => {
    if (s.wheelTimer !== undefined) clearTimeout(s.wheelTimer)
    s.wheelTimer = undefined
  }

  s.scrollOrigin = { x: 0, y: 0 }

  s.scrollLayout = ''

  s.bakedScale = 1

  s.paintedScroll = { left: 0, top: 0 }

  s.cameraDirty = true

  s.previewWheelDelta = 0

  s.previewWheelAt = 0

  s.previewWheelStep = -Infinity

  // Commit preview browsing to the shelf on close, so its SVG and index stay still underneath.

  // Commit preview browsing to the shelf on close, so its SVG and index stay still underneath.
  s.previewIndex = -1

  s.matches = () =>
    (s.visibleEntries ??= s.entries.filter(
      entry =>
        entry.style.display !== 'none' &&
        !entry.hasAttribute('data-topic-hidden') &&
        !entry.closest('.bookcase-index-group[hidden], .bookcase-index-group[data-topic-hidden]'),
    ))

  s.maxScale = () => Math.max(2, s.fitScale * 4)

  s.clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

  s.bound = (view: Camera) => {
    const insetX = Math.min(s.viewportWidth, s.box.width * view.scale) * 0.3
    const insetY = Math.min(s.viewportHeight, s.box.height * view.scale) * 0.3
    view.x = s.clamp(view.x, insetX - s.box.width * view.scale, s.viewportWidth - insetX)
    view.y = s.clamp(view.y, insetY - s.box.height * view.scale, s.viewportHeight - insetY)
  }

  s.paint = (now = performance.now()) => {
    s.frame = 0
    if (s.signal.aborted || s.dialog.open) return
    if (!s.cameraDirty && !s.aim) {
      s.camera.x = s.scrollOrigin.x - s.viewport.scrollLeft
      s.camera.y = s.scrollOrigin.y - s.viewport.scrollTop
    }
    if (s.aim) {
      s.cameraDirty = true
      // Each frame closes most of the gap (about 95% in 270 ms), so a new aim mid-glide carries
      // on from where the view is. Zoom runs in log scale about the viewport centre, so going from
      // the whole case to a shelf reads as one dolly-in rather than a slide and a jump.
      s.bound(s.aim)
      const k = 1 - Math.exp(-Math.min(64, Math.max(0, now - s.steppedAt)) / GLIDE_MS)
      s.steppedAt = now
      const cx = s.viewportWidth / 2
      const cy = s.viewportHeight / 2
      const scale = s.camera.scale * (s.aim.scale / s.camera.scale) ** k
      const fromX = (cx - s.camera.x) / s.camera.scale
      const fromY = (cy - s.camera.y) / s.camera.scale
      const x = fromX + ((cx - s.aim.x) / s.aim.scale - fromX) * k
      const y = fromY + ((cy - s.aim.y) / s.aim.scale - fromY) * k
      s.camera = { x: cx - x * scale, y: cy - y * scale, scale }
      if (
        Math.abs(scale / s.aim.scale - 1) < 0.002 &&
        Math.hypot(s.camera.x - s.aim.x, s.camera.y - s.aim.y) < 0.5
      ) {
        s.camera = s.aim
        s.aim = undefined
      } else s.schedule()
    }
    if (s.cameraDirty) s.bound(s.camera)
    const elapsed = now - s.painted.at
    if (elapsed > 0 && elapsed < 100 && s.camera.scale === s.painted.scale) {
      s.velocity = {
        x: s.velocity.x * 0.4 + ((s.camera.x - s.painted.x) / elapsed) * 0.6,
        y: s.velocity.y * 0.4 + ((s.camera.y - s.painted.y) / elapsed) * 0.6,
      }
    } else s.velocity = { x: 0, y: 0 }
    const moved =
      s.camera.x !== s.painted.x || s.camera.y !== s.painted.y || s.camera.scale !== s.painted.scale
    s.painted = { ...s.camera, at: now }
    if (!s.aim && s.scaleTimer === undefined && Math.abs(s.camera.scale - s.bakedScale) > 0.001)
      s.bakeScale(s.camera.scale)
    const layout = `${s.viewportWidth}:${s.viewportHeight}:${s.box.width}:${s.box.height}:${s.camera.scale}:${s.bakedScale}`
    if (s.cameraDirty && layout !== s.scrollLayout) {
      s.scrollLayout = layout
      s.scrollOrigin = {
        x: s.viewportWidth - Math.min(s.viewportWidth, s.box.width * s.camera.scale) * 0.3,
        y: s.viewportHeight - Math.min(s.viewportHeight, s.box.height * s.camera.scale) * 0.3,
      }
      const px = Math.min(128 * s.camera.scale, s.scrollOrigin.x)
      const py = Math.min(128 * s.camera.scale, s.scrollOrigin.y)
      s.scrollSpace.style.width = `${s.box.width * s.camera.scale + s.scrollOrigin.x * 2}px`
      s.scrollSpace.style.height = `${s.box.height * s.camera.scale + s.scrollOrigin.y * 2}px`
      s.pan.style.left = `${s.scrollOrigin.x - px}px`
      s.pan.style.top = `${s.scrollOrigin.y - py}px`
      s.pan.style.width = `${s.box.width * s.camera.scale + px * 2}px`
      s.pan.style.height = `${s.box.height * s.camera.scale + py * 2}px`
      s.world.style.left = `${px}px`
      s.world.style.top = `${py}px`
      s.applyWorldScale()
    }
    if (s.cameraDirty) {
      const left = s.scrollOrigin.x - s.camera.x
      const top = s.scrollOrigin.y - s.camera.y
      if (Math.abs(s.viewport.scrollLeft - left) > 0.5) s.viewport.scrollLeft = left
      if (Math.abs(s.viewport.scrollTop - top) > 0.5) s.viewport.scrollTop = top
      s.paintedScroll = { left: s.viewport.scrollLeft, top: s.viewport.scrollTop }
      s.cameraDirty = false
    }
    s.refreshShelfCovers()
    if (moved) s.schedulePrefetch()
    if (s.zoomOut) s.zoomOut.disabled = s.camera.scale <= s.minScale + 0.001
    if (s.zoomIn) s.zoomIn.disabled = s.camera.scale >= s.maxScale() - 0.001
  }
}
