import type { BookcaseScene } from './scene'
import {
  addArtwork,
  addCover,
  cancelSpineDetail,
  removeArtwork,
  removeExposedCover,
  setSpineResolution,
  SVG_NS,
} from './artwork'
import { SETTLE_MS } from './selection'
import { PULL_MS, type LiftGeometry } from './state'

export function installLift(s: BookcaseScene) {
  // Without motion everything settles at once: no copy slides home and no cover fades.
  s.dropLift = function dropLift(): void {
    s.pull?.cancel()
    s.pull = undefined
    s.reveal = undefined
    for (const { svg, cover } of s.settling) {
      for (const image of svg.querySelectorAll<SVGImageElement>('.bc-spine-art'))
        cancelSpineDetail(image)
      svg.remove()
      removeExposedCover(cover)
    }
    s.settling.clear()
    const spine = s.lifted?.querySelector<SVGImageElement>('.bc-spine-art')
    if (spine) cancelSpineDetail(spine)
    s.lifted?.remove()
    removeExposedCover(s.exposedCover)
    s.lifted = undefined
    s.exposedCover = undefined
    s.liftGeometry = undefined
    for (const book of s.books.values()) book.removeAttribute('data-lifted')
  }

  s.placeLift = function placeLift(): void {
    if (!s.lifted || !s.liftGeometry) return
    const { x, y, width, height, px, py } = s.liftGeometry
    const k = s.bakedScale
    s.lifted.style.left = `${x * k}px`
    s.lifted.style.top = `${y * k}px`
    s.lifted.style.width = `${width * k}px`
    s.lifted.style.height = `${height * k}px`
    s.lifted.style.transform = `translate(${px * k}px, ${py * k}px)`
  }

  s.applyWorldScale = function applyWorldScale(): void {
    const ratio = s.bakedScale > 0 ? s.camera.scale / s.bakedScale : 1
    if (Math.abs(ratio - 1) < 0.001) {
      s.world.style.transform = ''
      s.world.style.willChange = 'auto'
    } else {
      s.world.style.transform = `scale(${ratio})`
      s.world.style.willChange = 'transform'
    }
  }

  s.bakeScale = function bakeScale(scale: number): void {
    if (!(scale > 0) || Math.abs(scale - s.bakedScale) < 0.001) return
    s.bakedScale = scale
    const width = s.box.width * s.bakedScale
    const height = s.box.height * s.bakedScale
    s.scene.style.width = `${width}px`
    s.scene.style.height = `${height}px`
    s.world.style.width = `${width}px`
    s.world.style.height = `${height}px`
    s.scrollLayout = ''
    s.applyWorldScale()
    s.placeLift()
  }
}

export function mountLift(s: BookcaseScene) {
  s.fit = () => {
    s.cancelWheel()
    if (s.scaleTimer !== undefined) {
      clearTimeout(s.scaleTimer)
      s.scaleTimer = undefined
    }
    s.cameraDirty = true
    s.fitted = true
    s.aim = undefined
    const width = s.box.width * s.fitScale
    s.camera = {
      // Bays past the stage's width start at its left edge.
      x: width <= s.viewportWidth ? (s.viewportWidth - width) / 2 : 16,
      y: (s.viewportHeight - s.box.height * s.fitScale) / 2,
      scale: s.fitScale,
    }
    if (s.frame) {
      cancelAnimationFrame(s.frame)
      s.frame = 0
    }
    s.paint()
  }

  s.zoom = (factor: number, x = s.viewportWidth / 2, y = s.viewportHeight / 2) => {
    s.cancelWheel()
    s.cameraDirty = true
    const scale = s.clamp(s.camera.scale * factor, s.minScale, s.maxScale())
    const ratio = scale / s.camera.scale
    s.aim = undefined
    s.camera = { x: x - (x - s.camera.x) * ratio, y: y - (y - s.camera.y) * ratio, scale }
    s.fitted = false
    if (s.scaleTimer !== undefined) clearTimeout(s.scaleTimer)
    s.scaleTimer = window.setTimeout(() => {
      s.scaleTimer = undefined
      s.cameraDirty = true
      s.schedule()
    }, 120)
    s.schedule()
  }

  s.cloneBook = (index: number): SVGSVGElement | undefined => {
    const book = s.books.get(index)
    if (!book) return
    const bbox = s.bookBounds.get(index)
    if (!bbox) return
    const svg = document.createElementNS(SVG_NS, 'svg')
    svg.classList.add('bc-lift')
    svg.setAttribute('aria-hidden', 'true')
    svg.setAttribute('viewBox', `${bbox.x - 4} ${bbox.y - 4} ${bbox.width + 8} ${bbox.height + 8}`)
    const clone = book.cloneNode(true)
    if (!(clone instanceof SVGGElement)) return
    clone.removeAttribute('data-lifted')
    for (const image of clone.querySelectorAll<SVGImageElement>('.bc-cover, .bc-spine-art'))
      removeArtwork(image)
    addCover(clone, true)
    addArtwork(clone, 'spine', true)
    svg.append(clone)
    return svg
  }

  s.liftedIndex = (svg: SVGSVGElement | undefined) =>
    Number(svg?.querySelector<SVGGElement>('.bc-book')?.dataset.bookIndex ?? -1)

  // Without motion everything settles at once: no copy slides home and no cover fades.

  // A copy slides home along its pull and only then shows on the shelf again; the cover it

  // A copy slides home along its pull and only then shows on the shelf again; the cover it
  // uncovered goes with it.

  // A copy slides home along its pull and only then shows on the shelf again; the cover it
  // uncovered goes with it.
  s.sendHome = (
    previous: SVGSVGElement,
    { px, py }: LiftGeometry,
    cover: SVGImageElement | undefined,
  ) => {
    const home = s.books.get(s.liftedIndex(previous))
    const spine = previous.querySelector<SVGImageElement>('.bc-spine-art')
    if (spine) setSpineResolution(spine, false)
    let finished = false
    const settle = {
      svg: previous,
      cover,
      done: () => {
        if (finished) return
        finished = true
        s.settling.delete(settle)
        for (const image of previous.querySelectorAll<SVGImageElement>('.bc-spine-art'))
          cancelSpineDetail(image)
        previous.remove()
        removeExposedCover(cover)
        if (home && s.liftedIndex(s.lifted) !== s.liftedIndex(previous)) {
          home.removeAttribute('data-lifted')
          s.coverWindow = ''
          if (!s.signal.aborted) s.schedule()
        }
      },
    }
    if (s.reducedMotion.matches) return settle.done()
    while (s.settling.size >= 2) {
      const oldest = s.settling.values().next().value
      if (!oldest) break
      oldest.done()
    }
    s.settling.add(settle)
    const k = s.bakedScale
    previous
      .animate([{ translate: '0px 0px' }, { translate: `${-px * k}px ${-py * k}px` }], {
        duration: SETTLE_MS,
        easing: 'cubic-bezier(0.4, 0, 1, 1)',
        fill: 'forwards',
      })
      .finished.then(settle.done, settle.done)
  }

  // Puts the lifted copy back on its shelf without lifting another.

  // Puts the lifted copy back on its shelf without lifting another.
  s.lower = () => {
    const previous = s.lifted
    const geometry = s.liftGeometry
    const cover = s.exposedCover
    s.pull?.cancel()
    s.pull = undefined
    s.reveal = undefined
    s.lifted = undefined
    s.exposedCover = undefined
    s.liftGeometry = undefined
    if (previous && geometry) s.sendHome(previous, geometry, cover)
  }

  // The lift in three beats: the last copy slides home along its pull and only then shows on the

  // The lift in three beats: the last copy slides home along its pull and only then shows on the
  // shelf again; the new copy slides out from its slot; the cover it uncovers fades in after it.

  // The lift in three beats: the last copy slides home along its pull and only then shows on the
  // shelf again; the new copy slides out from its slot; the cover it uncovers fades in after it.
  s.lift = (index: number) => {
    const book = s.books.get(index)
    const svg = book && s.cloneBook(index)
    if (!book || !svg) return
    const still = s.reducedMotion.matches
    const previous = s.lifted
    const previousCover = s.exposedCover
    const previousGeometry = s.liftGeometry
    s.pull?.cancel()
    s.pull = undefined
    s.reveal = undefined
    s.lifted = undefined
    s.exposedCover = undefined
    const prevSibling = book.previousElementSibling
    const neighbor =
      prevSibling instanceof SVGGElement && prevSibling.classList.contains('bc-book')
        ? prevSibling
        : undefined
    // The cover the last copy uncovered stays if the new copy uncovers the same book.
    const kept = previousCover && neighbor?.contains(previousCover) ? previousCover : undefined
    if (previous && previousGeometry)
      s.sendHome(previous, previousGeometry, kept ? undefined : previousCover)
    s.world.append(svg)
    const liftedSpine = svg.querySelector<SVGImageElement>('.bc-spine-art')
    if (liftedSpine) setSpineResolution(liftedSpine, true)
    book.setAttribute('data-lifted', '')
    if (kept) s.exposedCover = kept
    else if (neighbor && !neighbor.querySelector('.bc-cover')) {
      addCover(neighbor)
      s.exposedCover = neighbor.querySelector<SVGImageElement>('.bc-cover') ?? undefined
      s.reveal = s.exposedCover?.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: still ? 0 : 220,
        delay: still ? 0 : PULL_MS * 0.45,
        easing: 'ease-out',
        fill: 'backwards',
      })
    }
    s.lifted = svg
    const bounds = svg.viewBox.baseVal
    const [dx, dy] = (book.dataset.offset ?? '0 0').split(' ').map(Number)
    const style = book.style
    s.liftGeometry = {
      x: bounds.x + dx - s.originX,
      y: bounds.y + dy - s.originY,
      width: bounds.width,
      height: bounds.height,
      px: parseFloat(style.getPropertyValue('--pull-x')),
      py: parseFloat(style.getPropertyValue('--pull-y')),
    }
    s.placeLift()
    if (still) return
    const { px, py } = s.liftGeometry
    const k = s.bakedScale
    s.pull = svg.animate([{ translate: `${-px * k}px ${-py * k}px` }, { translate: '0px 0px' }], {
      duration: PULL_MS,
      easing: 'cubic-bezier(0.2, 0, 0, 1)',
      fill: 'backwards',
    })
  }
}
