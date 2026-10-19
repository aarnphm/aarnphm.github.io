import { toHtml } from 'hast-util-to-html'
import { BOOKCASE_DEFAULTS, packShelves, type ShelfGroup } from '../../util/base/bookcase/geometry'
import { bookcaseScene, type SceneBook } from '../../util/base/bookcase/scene'
import {
  BOOKCASE_STYLES,
  SHELF_MATERIALS,
  type TopicShelf,
  placeShelves,
  stylePackOptions,
} from '../../util/base/bookcase/styles'
import { createHardcoverPreview } from './base-bookcase-preview'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const SVG_NS = 'http://www.w3.org/2000/svg'
const PRELOAD = 3
const MAX_SHELF_COVERS = 32
const MAX_DETAIL_SPINES = 4
const DETAIL_ENTER_PX = 360
const DETAIL_LEAVE_PX = 280
const COVER_MARGIN = 120

type Camera = { x: number; y: number; scale: number }
type LiftGeometry = { x: number; y: number; width: number; height: number; px: number; py: number }

// Camera glide time constant, and the lift's beats: the pull out of the slot, its wait for the
// glide, and the slide home of the copy it replaces.
const GLIDE_MS = 90
const PULL_MS = 280
const PULL_DELAY_MS = 160
const SETTLE_MS = 180
type Drag = { id: number; x: number; y: number; camera: Camera; book: number; moved: boolean }
type Cover = {
  url: string
  spine?: string
  ink?: string
  art?: string
  artDetail?: string
  artDetailWidth?: string
  artDetailHeight?: string
}

function retainCovers(root: HTMLElement, covers: Map<string, Cover>): void {
  const nodes = root.querySelectorAll<HTMLElement | SVGElement>('[data-bookcase-cover]')
  for (const node of nodes) {
    const slug = node.dataset.bookcaseCover
    const url = node instanceof HTMLImageElement ? node.getAttribute('src') : node.dataset.cover
    if (!slug || !url?.includes('/static/bookcase/')) continue
    const cover = { ...covers.get(slug), url }
    if (node.matches('.bc-book')) {
      cover.spine = node.style.getPropertyValue('--spine')
      cover.ink = node.style.getPropertyValue('--spine-ink')
      if (node.dataset.spineArt) cover.art = node.dataset.spineArt
      if (node.dataset.spineArtDetail) {
        cover.artDetail = node.dataset.spineArtDetail
        cover.artDetailWidth = node.dataset.spineArtDetailWidth
        cover.artDetailHeight = node.dataset.spineArtDetailHeight
      }
    }
    covers.set(slug, cover)
  }
  for (const node of nodes) {
    const slug = node.dataset.bookcaseCover
    const cover = slug ? covers.get(slug) : undefined
    if (!cover) continue
    if (node instanceof HTMLImageElement) node.src = cover.url
    else node.dataset.cover = cover.url
    if (cover.art) node.dataset.spineArt = cover.art
    if (cover.artDetail) {
      node.dataset.spineArtDetail = cover.artDetail
      node.dataset.spineArtDetailWidth = cover.artDetailWidth ?? ''
      node.dataset.spineArtDetailHeight = cover.artDetailHeight ?? ''
    }
    if (node.hasAttribute('data-spine-sampled')) {
      if (cover.spine) node.style.setProperty('--spine', cover.spine)
      if (cover.ink) node.style.setProperty('--spine-ink', cover.ink)
    }
  }
}

type ArtworkFace = 'cover' | 'spine'

// Each resident face has one SVG image. A temporary decoder keeps its low texture visible
// until the detail is ready; eviction or a new tier request cancels that decoder.
const spineDecoders = new WeakMap<SVGImageElement, HTMLImageElement>()
function cancelSpineDetail(image: SVGImageElement): void {
  delete image.dataset.spineResolutionPending
  const decoder = spineDecoders.get(image)
  if (!decoder) return
  spineDecoders.delete(image)
  decoder.removeAttribute('src')
}

function setSpineResolution(image: SVGImageElement, detail: boolean): void {
  const book = image.closest<SVGGElement>('.bc-book')
  const low = book?.dataset.spineArt
  const url = detail ? book?.dataset.spineArtDetail : low
  if (!book || !low || !url) return
  if (detail && book.dataset.spineArtDetailFailed === url) return
  const pending = spineDecoders.get(image)
  if (pending?.getAttribute('src') === url) return
  cancelSpineDetail(image)
  if (image.getAttribute('href') === url) return
  if (!detail) {
    image.setAttribute('href', low)
    image.dataset.spineResolution = 'low'
    return
  }
  const decoder = new Image()
  decoder.decoding = 'async'
  spineDecoders.set(image, decoder)
  image.dataset.spineResolutionPending = 'detail'
  decoder.src = url
  void decoder.decode().then(
    () => {
      if (spineDecoders.get(image) !== decoder) return
      spineDecoders.delete(image)
      delete image.dataset.spineResolutionPending
      if (image.isConnected) {
        image.setAttribute('href', url)
        image.dataset.spineResolution = 'detail'
      }
      decoder.removeAttribute('src')
    },
    () => {
      if (spineDecoders.get(image) !== decoder) return
      spineDecoders.delete(image)
      delete image.dataset.spineResolutionPending
      book.dataset.spineArtDetailFailed = url
      decoder.removeAttribute('src')
    },
  )
}

function removeArtwork(image: SVGImageElement): void {
  if (image.classList.contains('bc-spine-art')) {
    const book = image.closest<SVGGElement>('.bc-book')
    if (book) delete book.dataset.spineArtReady
    cancelSpineDetail(image)
  }
  image.remove()
}

function addArtwork(
  book: SVGGElement,
  face: ArtworkFace,
  selected = false,
): SVGImageElement | undefined {
  const spine = face === 'spine'
  const cover = spine ? book.dataset.spineArt : book.dataset.cover
  const transform = spine ? book.dataset.spineTransform : book.dataset.coverTransform
  const failed = spine ? book.dataset.spineArtFailed : book.dataset.coverFailed
  const side = book.querySelector(spine ? '.bc-face-spine' : '.bc-face-cover')
  const existing = book.querySelector<SVGImageElement>(spine ? '.bc-spine-art' : '.bc-cover')
  if (existing) return existing
  // A book turned 90 degrees shows its back cover, which has no transform and no image.
  if (
    !cover ||
    !side ||
    !transform ||
    failed === cover ||
    (!selected && !cover.includes(spine ? '/static/bookcase-spines/' : '/static/bookcase/'))
  )
    return
  const [width, height] = (
    (spine ? book.dataset.spineSize : book.dataset.coverSize) ?? '0 0'
  ).split(' ')
  const image = document.createElementNS(SVG_NS, 'image')
  image.classList.add(spine ? 'bc-spine-art' : 'bc-cover')
  image.setAttribute('href', cover)
  image.setAttribute('width', width)
  image.setAttribute('height', height)
  image.setAttribute('preserveAspectRatio', spine ? 'none' : 'xMidYMid meet')
  image.setAttribute('decoding', 'async')
  image.setAttribute('transform', transform)
  if (spine) {
    image.dataset.spineResolution = 'low'
    image.addEventListener('load', () => {
      if (image.isConnected) book.dataset.spineArtReady = ''
    })
  }
  image.addEventListener('error', () => {
    if (spine && image.dataset.spineResolution === 'detail') {
      book.dataset.spineArtDetailFailed = book.dataset.spineArtDetail
      setSpineResolution(image, false)
      return
    }
    if (spine) book.dataset.spineArtFailed = cover
    else book.dataset.coverFailed = cover
    removeArtwork(image)
  })
  side.after(image)
  return image
}

const addCover = (book: SVGGElement, selected = false) => addArtwork(book, 'cover', selected)

// The camera owns resident images; a lift may also reveal a temporary neighbour cover.
function removeExposedCover(image: SVGImageElement | undefined): void {
  if (image && !image.hasAttribute('data-shelf-resident')) image.remove()
}

function initializeBookcase(root: HTMLElement, covers: Map<string, Cover>): void {
  if (root.dataset.bookcaseReady !== undefined) return
  const scene = root.querySelector<SVGSVGElement>('svg.bookcase-scene')
  const world = root.querySelector<HTMLElement>('.bookcase-world')
  const viewport = root.querySelector<HTMLElement>('.bookcase-viewport')
  const indexPane = root.querySelector<HTMLElement>('.bookcase-index')
  const dialog = root.querySelector<HTMLDialogElement>('.bookcase-preview')
  if (!scene || !world || !viewport || !indexPane || !dialog) return
  retainCovers(root, covers)
  root.dataset.bookcaseReady = ''
  const signal = rootNavSignal(root)
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  const entries = [...root.querySelectorAll<HTMLLIElement>('.bookcase-entry')]
  const groups = [...root.querySelectorAll<HTMLDetailsElement>('.bookcase-index-group')]
  const books = new Map<number, SVGGElement>()
  const bookBounds = new Map<number, DOMRect>()
  const spineLengths = new Map<number, number>()
  const spineBounds = new Map<number, { x: number; y: number; width: number; height: number }>()
  const cacheBookGeometry = (index: number, book: SVGGElement) => {
    bookBounds.set(index, book.getBBox())
    const transform = book.dataset.spineTransform
    const [width, height] = (book.dataset.spineSize ?? '').split(' ').map(Number)
    if (!transform || !Number.isFinite(width) || !Number.isFinite(height)) return
    const values = transform.slice(7, -1).trim().split(/[ ,]+/).map(Number)
    if (values.length !== 6 || !values.every(Number.isFinite)) return
    const [a, b, c, d, e, f] = values
    spineLengths.set(index, Math.hypot(c, d) * height)
    const xs = [0, a * width, c * height, a * width + c * height]
    const ys = [0, b * width, d * height, b * width + d * height]
    spineBounds.set(index, {
      x: e + Math.min(...xs),
      y: f + Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
    })
  }
  const entryByIndex = new Map(entries.map(entry => [Number(entry.dataset.bookIndex), entry]))
  for (const book of scene.querySelectorAll<SVGGElement>('.bc-book'))
    books.set(Number(book.dataset.bookIndex), book)
  const residentCovers = new Map<string, SVGImageElement>()
  let coverWindow = ''
  let [originX, originY] = (scene.dataset.origin ?? '0 0').split(' ').map(Number)
  let box = scene.viewBox.baseVal
  for (const [index, book] of books) cacheBookGeometry(index, book)
  const inventory = new Map(
    [...books].map(([index, book]) => [
      index,
      {
        path: book.dataset.bookPath ?? '',
        fullCover: book.dataset.fullCover,
        item: {
          index,
          slug: book.dataset.baseSlug ?? '',
          title: book.dataset.title ?? '',
          author: book.dataset.author,
          spine: book.style.getPropertyValue('--spine'),
          ink: book.style.getPropertyValue('--spine-ink'),
          cover: book.dataset.cover,
          coverSlug: book.dataset.bookcaseCover,
          sampleSpine: false,
          spineArt: book.dataset.spineArt,
          spineArtDetail: book.dataset.spineArtDetail,
          spineArtDetailWidth: Number(book.dataset.spineArtDetailWidth) || undefined,
          spineArtDetailHeight: Number(book.dataset.spineArtDetailHeight) || undefined,
        },
      },
    ]),
  )
  // The chosen topic (a group, a tag, "other", or "" for all) lives in the page URL as ?topic=.
  let topic = ''
  const topicShelves: Record<string, TopicShelf> = JSON.parse(root.dataset.topicShelves ?? '{}')
  let arrangement = `|${entries.map(entry => entry.dataset.bookIndex).join(',')}`

  const arrange = (visible: HTMLLIElement[]) => {
    const key = `${topic}|${visible.map(entry => entry.dataset.bookIndex).join(',')}`
    if (key === arrangement) return false
    arrangement = key
    const sections = new Map<Element | string | null, ShelfGroup<SceneBook & { index: number }>>()
    for (const entry of visible) {
      const index = Number(entry.dataset.bookIndex)
      const book = inventory.get(index)
      if (!book) continue
      // Inside a topic, shelves group by the topic's own property instead of the index groups.
      const section = entry.closest('.bookcase-index-group')
      const slot = topic ? (entry.dataset.topicShelf ?? null) : section
      let group = sections.get(slot)
      if (!group) {
        const label = topic
          ? entry.dataset.topicShelf
          : section?.querySelector('.bookcase-index-label')?.textContent
        group = { label: label ?? null, books: [] }
        sections.set(slot, group)
      }
      group.books.push(book)
    }
    const shelves = [...sections.values()]
    if (topic)
      shelves.sort((a, b) =>
        a.label === null ? 1 : b.label === null ? -1 : a.label.localeCompare(b.label),
      )
    const own = topic ? topicShelves[topic] : undefined
    const style =
      own?.style ?? BOOKCASE_STYLES.find(name => name === root.dataset.shelfStyle) ?? 'straight'
    const material = own
      ? own.material
      : SHELF_MATERIALS.find(name => name === root.dataset.shelfMaterial)
    const options = {
      bookHeight: Number(root.dataset.bookHeight) || BOOKCASE_DEFAULTS.bookHeight,
      shelfWidth: Number(root.dataset.shelfWidth) || BOOKCASE_DEFAULTS.shelfWidth,
      // A topic's shelves group by small subcategories, so they share runs.
      continuous: Boolean(topic),
    }
    const layout = placeShelves(
      packShelves(shelves, stylePackOptions(style, options, visible.length)),
      style,
      material,
      { showTags: own?.showTags ?? root.dataset.showTags !== 'false' },
    )
    for (const book of layout.books) book.index = book.item.index
    const name = root.closest<HTMLElement>('.base-view')?.dataset.baseViewName ?? 'bookcase'
    const template = document.createElement('template')
    template.innerHTML = toHtml(
      bookcaseScene(
        layout,
        `${name}: ${layout.books.length} books on ${layout.shelves.length} shelves`,
      ),
    )
    const next = template.content.querySelector('svg')
    if (!next) return false
    dropLift()
    for (const image of residentCovers.values()) removeArtwork(image)
    for (const attribute of next.attributes) scene.setAttribute(attribute.name, attribute.value)
    scene.replaceChildren(...next.childNodes)
    ;[originX, originY] = (scene.dataset.origin ?? '0 0').split(' ').map(Number)
    box = scene.viewBox.baseVal
    books.clear()
    bookBounds.clear()
    spineLengths.clear()
    spineBounds.clear()
    residentCovers.clear()
    coverWindow = ''
    for (const book of scene.querySelectorAll<SVGGElement>('.bc-book')) {
      const index = Number(book.dataset.bookIndex)
      const fullCover = inventory.get(index)?.fullCover
      if (fullCover) book.dataset.fullCover = fullCover
      books.set(index, book)
      cacheBookGeometry(index, book)
    }
    fitted = true
    resize()
    return true
  }

  const readout = root.querySelector<HTMLElement>('[data-bookcase-readout]')
  const shelfLabel = root.querySelector<HTMLElement>('[data-bookcase-shelf]')
  const previewArt = dialog.querySelector<HTMLElement>('.bookcase-preview-art')
  const hardcover = previewArt ? createHardcoverPreview(previewArt, signal) : undefined
  const previousButton = dialog.querySelector<HTMLButtonElement>('[data-preview="previous"]')
  const nextButton = dialog.querySelector<HTMLButtonElement>('[data-preview="next"]')
  const zoomOut = root.querySelector<HTMLButtonElement>('[data-camera="out"]')
  const zoomIn = root.querySelector<HTMLButtonElement>('[data-camera="in"]')
  const preloaded = new Map<string, HTMLImageElement>()
  let camera: Camera = { x: 0, y: 0, scale: 1 }
  let viewportWidth = viewport.clientWidth
  let viewportHeight = viewport.clientHeight
  let fitScale = 1
  let fitted = true
  let focus = -1
  let lifted: SVGSVGElement | undefined
  let exposedCover: SVGImageElement | undefined
  let liftGeometry: LiftGeometry | undefined
  // The pull of the lifted copy, and copies still sliding home with the covers they exposed.
  let pull: Animation | undefined
  let reveal: Animation | undefined
  const settling = new Set<{ svg: SVGSVGElement; cover?: SVGImageElement }>()
  // A camera the view glides toward, and when it last stepped.
  let aim: Camera | undefined
  let steppedAt = 0
  let drag: Drag | undefined
  let frame = 0
  let indexFrame = 0
  let followIndexScroll = false
  let browsePosition = 0
  const scrollStep = Math.max(1, Number(root.dataset.scrollStep) || 56)
  let previewWheelDelta = 0
  let previewWheelAt = 0
  let previewWheelStep = -Infinity
  let returnFocus: HTMLElement | undefined
  let searchOpen: Map<HTMLDetailsElement, boolean> | undefined

  const matches = () =>
    [...indexPane.querySelectorAll<HTMLLIElement>('.bookcase-entry')].filter(
      entry =>
        entry.style.display !== 'none' &&
        !entry.hasAttribute('data-topic-hidden') &&
        !entry.closest('.bookcase-index-group[hidden], .bookcase-index-group[data-topic-hidden]'),
    )
  const maxScale = () => Math.max(2, fitScale * 4)
  const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

  const bound = (view: Camera) => {
    const insetX = Math.min(viewportWidth, box.width * view.scale) * 0.3
    const insetY = Math.min(viewportHeight, box.height * view.scale) * 0.3
    view.x = clamp(view.x, insetX - box.width * view.scale, viewportWidth - insetX)
    view.y = clamp(view.y, insetY - box.height * view.scale, viewportHeight - insetY)
  }
  const paint = (now = performance.now()) => {
    frame = 0
    if (signal.aborted) return
    if (aim) {
      // Each frame closes most of the gap (about 95% in 270 ms), so a new aim mid-glide carries
      // on from where the view is. Zoom runs in log scale about the viewport centre, so going from
      // the whole case to a shelf reads as one dolly-in rather than a slide and a jump.
      bound(aim)
      const k = 1 - Math.exp(-Math.min(64, Math.max(0, now - steppedAt)) / GLIDE_MS)
      steppedAt = now
      const cx = viewportWidth / 2
      const cy = viewportHeight / 2
      const scale = camera.scale * (aim.scale / camera.scale) ** k
      const fromX = (cx - camera.x) / camera.scale
      const fromY = (cy - camera.y) / camera.scale
      const x = fromX + ((cx - aim.x) / aim.scale - fromX) * k
      const y = fromY + ((cy - aim.y) / aim.scale - fromY) * k
      camera = { x: cx - x * scale, y: cy - y * scale, scale }
      if (
        Math.abs(scale / aim.scale - 1) < 0.002 &&
        Math.hypot(camera.x - aim.x, camera.y - aim.y) < 0.5
      ) {
        camera = aim
        aim = undefined
      } else schedule()
    }
    bound(camera)
    world.style.width = `${box.width * camera.scale}px`
    world.style.height = `${box.height * camera.scale}px`
    world.style.transform = `translate(${camera.x}px, ${camera.y}px)`
    placeLift()
    refreshShelfCovers()
    if (zoomOut) zoomOut.disabled = camera.scale <= fitScale + 0.001
    if (zoomIn) zoomIn.disabled = camera.scale >= maxScale() - 0.001
  }
  function refreshShelfCovers(): void {
    const pixelRatio = Math.min(2, window.devicePixelRatio || 1)
    const key = `${focus}:${Math.round(camera.x / 64)}:${Math.round(camera.y / 64)}:${Math.round(camera.scale * 32)}:${viewportWidth}:${viewportHeight}:${pixelRatio}`
    if (key === coverWindow) return
    coverWindow = key
    const candidates: {
      index: number
      face: ArtworkFace
      key: string
      score: number
      detail?: boolean
    }[] = []
    for (const [index, book] of books) {
      const bounds = bookBounds.get(index)
      if (!bounds || book.hasAttribute('data-lifted')) continue
      const width = bounds.width * camera.scale
      const height = bounds.height * camera.scale
      const x = (bounds.x - originX) * camera.scale + camera.x
      const y = (bounds.y - originY) * camera.scale + camera.y
      if (
        height < 72 ||
        x + width < -COVER_MARGIN ||
        x > viewportWidth + COVER_MARGIN ||
        y + height < -COVER_MARGIN ||
        y > viewportHeight + COVER_MARGIN
      )
        continue
      const end = !book.nextElementSibling?.matches('.bc-book:not([data-lifted])')
      const distance =
        Math.abs(y + height / 2 - viewportHeight / 2) +
        Math.abs(x + width / 2 - viewportWidth / 2) * 0.25
      if (book.dataset.coverTransform && book.dataset.cover)
        candidates.push({
          index,
          face: 'cover',
          key: `${index}:cover`,
          score: distance * (end ? 0.25 : 1),
        })
      if (book.dataset.spineArt) {
        const key = `${index}:spine`
        const resident = residentCovers.get(key)
        const detailed =
          resident?.dataset.spineResolution === 'detail' ||
          Boolean(resident && spineDecoders.has(resident))
        const length = (spineLengths.get(index) ?? 0) * camera.scale * pixelRatio
        const spine = spineBounds.get(index)
        const sx = spine ? (spine.x - originX) * camera.scale + camera.x : -Infinity
        const sy = spine ? (spine.y - originY) * camera.scale + camera.y : -Infinity
        const visible =
          spine &&
          sx + spine.width * camera.scale >= 0 &&
          sx <= viewportWidth &&
          sy + spine.height * camera.scale >= 0 &&
          sy <= viewportHeight
        candidates.push({
          index,
          face: 'spine',
          key,
          score: distance * 0.5,
          detail: Boolean(
            visible &&
            book.dataset.spineArtDetail &&
            book.dataset.spineArtDetailFailed !== book.dataset.spineArtDetail &&
            length > (detailed ? DETAIL_LEAVE_PX : DETAIL_ENTER_PX),
          ),
        })
      }
    }
    const wanted = candidates.sort((a, b) => a.score - b.score).slice(0, MAX_SHELF_COVERS)
    const keys = new Set(wanted.map(candidate => candidate.key))
    for (const [key, image] of residentCovers) {
      if (keys.has(key) && image.isConnected) continue
      removeArtwork(image)
      residentCovers.delete(key)
    }
    // Downgrade outgoing detail slots before decoding replacements, including pending loads.
    const detailed = new Set(
      wanted
        .filter(candidate => candidate.detail)
        .slice(0, MAX_DETAIL_SPINES)
        .map(candidate => candidate.key),
    )
    for (const [key, image] of residentCovers) {
      if (image.classList.contains('bc-spine-art') && !detailed.has(key))
        setSpineResolution(image, false)
    }
    for (const { index, face, key } of wanted) {
      const book = books.get(index)
      let image = residentCovers.get(key)
      if (!image && book) {
        image = addArtwork(book, face)
        if (image) {
          image.dataset.shelfResident = ''
          residentCovers.set(key, image)
        }
      }
      if (image && face === 'spine') setSpineResolution(image, detailed.has(key))
    }
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(paint)
  }
  const fit = () => {
    fitted = true
    aim = undefined
    camera = {
      x: (viewportWidth - box.width * fitScale) / 2,
      y: (viewportHeight - box.height * fitScale) / 2,
      scale: fitScale,
    }
    schedule()
  }
  const zoom = (factor: number, x = viewportWidth / 2, y = viewportHeight / 2) => {
    const scale = clamp(camera.scale * factor, fitScale, maxScale())
    const ratio = scale / camera.scale
    aim = undefined
    camera = { x: x - (x - camera.x) * ratio, y: y - (y - camera.y) * ratio, scale }
    fitted = false
    schedule()
  }

  const cloneBook = (index: number): SVGSVGElement | undefined => {
    const book = books.get(index)
    if (!book) return
    const bbox = bookBounds.get(index)
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

  const liftedIndex = (svg: SVGSVGElement | undefined) =>
    Number(svg?.querySelector<SVGGElement>('.bc-book')?.dataset.bookIndex ?? -1)

  // Without motion everything settles at once: no copy slides home and no cover fades.
  function dropLift(): void {
    pull?.cancel()
    pull = undefined
    reveal = undefined
    for (const { svg, cover } of settling) {
      for (const image of svg.querySelectorAll<SVGImageElement>('.bc-spine-art'))
        cancelSpineDetail(image)
      svg.remove()
      removeExposedCover(cover)
    }
    settling.clear()
    const spine = lifted?.querySelector<SVGImageElement>('.bc-spine-art')
    if (spine) cancelSpineDetail(spine)
    lifted?.remove()
    removeExposedCover(exposedCover)
    lifted = undefined
    exposedCover = undefined
    liftGeometry = undefined
    for (const book of books.values()) book.removeAttribute('data-lifted')
  }

  // A copy slides home along its pull and only then shows on the shelf again; the cover it
  // uncovered goes with it.
  const sendHome = (
    previous: SVGSVGElement,
    { px, py }: LiftGeometry,
    cover: SVGImageElement | undefined,
  ) => {
    const home = books.get(liftedIndex(previous))
    const spine = previous.querySelector<SVGImageElement>('.bc-spine-art')
    if (spine) setSpineResolution(spine, false)
    const settle = { svg: previous, cover }
    const done = () => {
      settling.delete(settle)
      previous.remove()
      removeExposedCover(cover)
      if (home && liftedIndex(lifted) !== liftedIndex(previous)) {
        home.removeAttribute('data-lifted')
        coverWindow = ''
        if (!signal.aborted) schedule()
      }
    }
    if (reducedMotion.matches) return done()
    settling.add(settle)
    previous
      .animate(
        [
          { translate: '0px 0px' },
          { translate: `${-px * camera.scale}px ${-py * camera.scale}px` },
        ],
        { duration: SETTLE_MS, easing: 'cubic-bezier(0.4, 0, 1, 1)', fill: 'forwards' },
      )
      .finished.then(done, done)
  }

  // Puts the lifted copy back on its shelf without lifting another.
  const lower = () => {
    const previous = lifted
    const geometry = liftGeometry
    const cover = exposedCover
    pull?.cancel()
    pull = undefined
    reveal = undefined
    lifted = undefined
    exposedCover = undefined
    liftGeometry = undefined
    if (previous && geometry) sendHome(previous, geometry, cover)
  }

  // The lift in three beats: the last copy slides home along its pull and only then shows on the
  // shelf again; the new copy slides out from its slot; the cover it uncovers fades in after it.
  const lift = (index: number) => {
    const book = books.get(index)
    const svg = book && cloneBook(index)
    if (!book || !svg) return
    const still = reducedMotion.matches
    const previous = lifted
    const previousCover = exposedCover
    const previousGeometry = liftGeometry
    pull?.cancel()
    pull = undefined
    reveal = undefined
    lifted = undefined
    exposedCover = undefined
    const neighbors = [...(book.parentElement?.querySelectorAll<SVGGElement>('.bc-book') ?? [])]
    const neighbor = neighbors[neighbors.indexOf(book) - 1]
    // The cover the last copy uncovered stays if the new copy uncovers the same book.
    const kept = previousCover && neighbor?.contains(previousCover) ? previousCover : undefined
    if (previous && previousGeometry)
      sendHome(previous, previousGeometry, kept ? undefined : previousCover)
    world.append(svg)
    const liftedSpine = svg.querySelector<SVGImageElement>('.bc-spine-art')
    if (liftedSpine) setSpineResolution(liftedSpine, true)
    book.setAttribute('data-lifted', '')
    if (kept) exposedCover = kept
    else if (neighbor && !neighbor.querySelector('.bc-cover')) {
      addCover(neighbor)
      exposedCover = neighbor.querySelector<SVGImageElement>('.bc-cover') ?? undefined
      reveal = exposedCover?.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: still ? 0 : 220,
        delay: still ? 0 : PULL_MS * 0.45,
        easing: 'ease-out',
        fill: 'backwards',
      })
    }
    lifted = svg
    const bounds = svg.viewBox.baseVal
    const [dx, dy] = (book.dataset.offset ?? '0 0').split(' ').map(Number)
    const style = book.style
    liftGeometry = {
      x: bounds.x + dx - originX,
      y: bounds.y + dy - originY,
      width: bounds.width,
      height: bounds.height,
      px: parseFloat(style.getPropertyValue('--pull-x')),
      py: parseFloat(style.getPropertyValue('--pull-y')),
    }
    placeLift()
    if (still) return
    const { px, py } = liftGeometry
    pull = svg.animate(
      [{ translate: `${-px * camera.scale}px ${-py * camera.scale}px` }, { translate: '0px 0px' }],
      { duration: PULL_MS, easing: 'cubic-bezier(0.2, 0, 0, 1)', fill: 'backwards' },
    )
  }

  function placeLift(): void {
    if (!lifted || !liftGeometry) return
    const { x, y, width, height, px, py } = liftGeometry
    lifted.style.left = `${x * camera.scale}px`
    lifted.style.top = `${y * camera.scale}px`
    lifted.style.width = `${width * camera.scale}px`
    lifted.style.height = `${height * camera.scale}px`
    lifted.style.transform = `translate(${px * camera.scale}px, ${py * camera.scale}px)`
  }

  const copyDetail = (row: HTMLLIElement, target: Element | null) => {
    const detail = row.querySelector('.bookcase-entry-detail')
    target?.replaceChildren(...[...(detail?.childNodes ?? [])].map(node => node.cloneNode(true)))
  }

  const select = (index: number, reveal = false) => {
    const row = entryByIndex.get(index)
    const book = books.get(index)
    if (!row || !book) return
    if (reveal) {
      const group = row.closest('details')
      if (group instanceof HTMLDetailsElement) group.open = true
      const bounds = row.getBoundingClientRect()
      const pane = indexPane.getBoundingClientRect()
      const top = bounds.top - pane.top - pane.height / 2 + bounds.height / 2
      followIndexScroll = false
      // Scroll before changing the drawing, so scrolling cannot force its new layout synchronously.
      indexPane.scrollBy({ top, behavior: reducedMotion.matches ? 'instant' : 'smooth' })
    }
    const previous = entryByIndex.get(focus)
    if (previous !== row) {
      previous?.removeAttribute('data-current')
      previous?.querySelector('.bookcase-entry-link')?.removeAttribute('aria-current')
    }
    focus = index
    row.setAttribute('data-current', '')
    row.querySelector('.bookcase-entry-link')?.setAttribute('aria-current', 'true')
    const groupLabel = row.closest('.bookcase-index-group')?.querySelector('.bookcase-index-label')
    if (shelfLabel)
      shelfLabel.textContent = (topic ? row.dataset.topicShelf : groupLabel?.textContent) ?? ''
    const visible = matches()
    browsePosition = visible.indexOf(row)
    if (readout)
      readout.textContent = `${String(visible.indexOf(row) + 1).padStart(3, '0')} / ${visible.length}`
    if (
      !lifted ||
      lifted.querySelector<SVGGElement>('.bc-book')?.dataset.bookIndex !== String(index)
    )
      lift(index)
    const urls = new Set<string>()
    for (let i = browsePosition - PRELOAD; i <= browsePosition + PRELOAD; i++) {
      const neighbor = visible[i]
      const url = neighbor
        ? books.get(Number(neighbor.dataset.bookIndex))?.dataset.cover
        : undefined
      if (url) urls.add(url)
    }
    for (const url of preloaded.keys()) if (!urls.has(url)) preloaded.delete(url)
    for (const url of urls) {
      if (preloaded.has(url)) continue
      const image = new Image()
      image.decoding = 'async'
      image.src = url
      preloaded.set(url, image)
    }
  }

  // Nothing is chosen until a click, the keyboard or the wheel picks a book; Escape, a click on
  // the empty stage or leaving the bookcase puts it back on its shelf.
  const deselect = () => {
    const row = entryByIndex.get(focus)
    row?.removeAttribute('data-current')
    row?.querySelector('.bookcase-entry-link')?.removeAttribute('aria-current')
    focus = -1
    lower()
    if (readout) readout.textContent = `– / ${matches().length}`
    if (shelfLabel) shelfLabel.textContent = ''
  }

  // A jump to a book glides; scroll-browsing passes `glide` false, since the wheel already moves
  // the view in small steps.
  const focusCamera = (index: number, towards = index, progress = 0, glide = true) => {
    const book = books.get(index)
    if (!book) return
    const scale = Math.max(
      fitScale,
      Math.min(1.25, viewportHeight / 600, viewportWidth / (box.width + 80)),
    )
    const startY = Number(book.dataset.cy)
    const endY = Number(books.get(towards)?.dataset.cy ?? startY)
    const y = startY + (endY - startY) * progress - originY
    const next = {
      x: (viewportWidth - box.width * scale) / 2,
      y: viewportHeight / 2 - y * scale,
      scale,
    }
    fitted = false
    if (glide && !reducedMotion.matches) {
      if (!aim) steppedAt = performance.now()
      aim = next
      // The book waits for the view to close in before it slides out.
      if (pull && Number(pull.currentTime ?? 0) < 16) {
        pull.effect?.updateTiming({ delay: PULL_DELAY_MS })
        reveal?.effect?.updateTiming({ delay: PULL_DELAY_MS + PULL_MS * 0.45 })
      }
    } else {
      aim = undefined
      camera = next
    }
    schedule()
  }

  const renderPreview = () => {
    const row = entryByIndex.get(focus)
    if (!row) return
    const visible = matches()
    const position = visible.indexOf(row)
    const link = row.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
    const previewLink = dialog.querySelector<HTMLAnchorElement>('.bookcase-preview-link')
    const title = row.querySelector('.bookcase-entry-title')?.textContent ?? ''
    dialog.setAttribute('aria-label', `Preview: ${title}`)
    const heading = dialog.querySelector('.bookcase-preview-title')
    if (heading) heading.textContent = title
    const byline = dialog.querySelector('.bookcase-preview-byline')
    if (byline) byline.textContent = row.querySelector('.bookcase-entry-byline')?.textContent ?? ''
    const positionLabel = dialog.querySelector('.bookcase-preview-position')
    if (positionLabel) positionLabel.textContent = `${position + 1} / ${visible.length}`
    if (previousButton) previousButton.disabled = position <= 0
    if (nextButton) nextButton.disabled = position < 0 || position >= visible.length - 1
    if (previewLink && link) {
      previewLink.href = link.href
      previewLink.dataset.slug = link.dataset.slug ?? ''
    }
    copyDetail(row, dialog.querySelector('.bookcase-preview-detail'))
    const book = books.get(focus)
    if (book) {
      const style = book.style
      void hardcover
        ?.show({
          cover: book.dataset.fullCover ?? book.dataset.cover,
          thumbnail: book.dataset.cover,
          spine: style.getPropertyValue('--spine').trim(),
          ink: style.getPropertyValue('--spine-ink').trim(),
          title,
          author: row.querySelector('.bookcase-entry-byline')?.textContent ?? '',
          spineArt: book.dataset.spineArt,
          spineArtDetail: book.dataset.spineArtDetail,
          spineArtDetailWidth: Number(book.dataset.spineArtDetailWidth) || undefined,
          spineArtDetailHeight: Number(book.dataset.spineArtDetailHeight) || undefined,
        })
        .then(ready => {
          if (ready && dialog.open) hardcover?.reveal()
        })
    }
  }

  const openPreview = (index: number) => {
    select(index, true)
    if (!dialog.open) {
      const active = document.activeElement
      returnFocus = active instanceof HTMLElement && active !== document.body ? active : undefined
      dialog.showModal()
    }
    renderPreview()
  }
  const browsePreview = (delta: number) => {
    const visible = matches()
    const position = visible.findIndex(entry => Number(entry.dataset.bookIndex) === focus)
    const row = visible[position + delta]
    if (!row) return
    select(Number(row.dataset.bookIndex), true)
    focusCamera(focus)
    renderPreview()
  }

  const wheelUnit = (event: WheelEvent, height: number) =>
    event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? height
        : 1

  viewport.addEventListener(
    'pointerdown',
    event => {
      if (!event.isPrimary || event.button !== 0) return
      const target = event.target
      const book = target instanceof Element ? target.closest<SVGGElement>('.bc-book') : null
      drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        camera: { ...camera },
        book: book ? Number(book.dataset.bookIndex) : -1,
        moved: false,
      }
      viewport.setPointerCapture(event.pointerId)
    },
    { signal },
  )
  viewport.addEventListener(
    'pointermove',
    event => {
      if (!drag || event.pointerId !== drag.id) return
      const dx = event.clientX - drag.x
      const dy = event.clientY - drag.y
      if (!drag.moved && Math.hypot(dx, dy) < 5) return
      drag.moved = true
      fitted = false
      viewport.dataset.dragging = ''
      aim = undefined
      camera = { ...drag.camera, x: drag.camera.x + dx, y: drag.camera.y + dy }
      schedule()
    },
    { signal },
  )
  viewport.addEventListener(
    'pointerup',
    event => {
      if (!drag || event.pointerId !== drag.id) return
      const gesture = drag
      drag = undefined
      delete viewport.dataset.dragging
      if (viewport.hasPointerCapture(event.pointerId))
        viewport.releasePointerCapture(event.pointerId)
      if (!gesture.moved && gesture.book >= 0) openPreview(gesture.book)
      else if (!gesture.moved && focus >= 0) deselect()
    },
    { signal },
  )
  const cancelDrag = () => {
    drag = undefined
    delete viewport.dataset.dragging
  }
  viewport.addEventListener('pointercancel', cancelDrag, { signal })
  viewport.addEventListener('lostpointercapture', cancelDrag, { signal })
  viewport.addEventListener(
    'keydown',
    event => {
      if (event.target !== viewport) return
      if (event.key === 'Home') fit()
      else if (event.key === '+' || event.key === '=') zoom(1.35)
      else if (event.key === '-') zoom(1 / 1.35)
      else if (event.key === 'Enter' && focus >= 0) openPreview(focus)
      else if (event.key === 'Escape' && focus >= 0) deselect()
      else if (event.key.startsWith('Arrow')) {
        aim = undefined
        if (event.key === 'ArrowLeft') camera.x += 48
        if (event.key === 'ArrowRight') camera.x -= 48
        if (event.key === 'ArrowUp') camera.y += 48
        if (event.key === 'ArrowDown') camera.y -= 48
        fitted = false
        schedule()
      } else return
      event.preventDefault()
    },
    { signal },
  )
  viewport.addEventListener(
    'wheel',
    event => {
      event.preventDefault()
      if (event.ctrlKey || event.metaKey) {
        const bounds = viewport.getBoundingClientRect()
        zoom(
          Math.exp(-event.deltaY * 0.01),
          event.clientX - bounds.left,
          event.clientY - bounds.top,
        )
      } else {
        const unit = wheelUnit(event, viewportHeight)
        if (Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey) {
          aim = undefined
          camera.x -= (event.deltaX || event.deltaY) * unit
          fitted = false
          schedule()
          return
        }
        const visible = matches()
        if (!visible.length) return
        const position = clamp(
          browsePosition + (event.deltaY * unit) / scrollStep,
          0,
          visible.length - 1,
        )
        const selected = Number(visible[Math.round(position)].dataset.bookIndex)
        if (selected !== focus) select(selected, true)
        browsePosition = position
        const start = Math.floor(position)
        focusCamera(
          Number(visible[start].dataset.bookIndex),
          Number(visible[Math.min(start + 1, visible.length - 1)].dataset.bookIndex),
          position - start,
          false,
        )
      }
    },
    { passive: false, signal },
  )

  root.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const cameraControl = event.target.closest<HTMLElement>('[data-camera]')
      if (cameraControl?.dataset.camera === 'fit') fit()
      else if (cameraControl) zoom(cameraControl.dataset.camera === 'in' ? 1.35 : 1 / 1.35)
      const action = event.target.closest<HTMLElement>('[data-preview]')?.dataset.preview
      if (action === 'close') dialog.close()
      else if (action === 'previous') browsePreview(-1)
      else if (action === 'next') browsePreview(1)
    },
    { signal },
  )
  // Leaving lowers the book, unless the keyboard is still inside the bookcase.
  root.addEventListener(
    'pointerleave',
    () => {
      if (focus < 0 || dialog.open || drag) return
      const active = document.activeElement
      if (active && root.contains(active) && active.matches(':focus-visible')) return
      deselect()
    },
    { signal },
  )
  root.addEventListener(
    'focusout',
    event => {
      const next = event.relatedTarget
      if (focus < 0 || dialog.open || (next instanceof Node && root.contains(next))) return
      if (!root.matches(':hover')) deselect()
    },
    { signal },
  )
  const selectFromIndex = (event: Event) => {
    if (!(event.target instanceof Element)) return
    const row = event.target.closest<HTMLLIElement>('.bookcase-entry')
    if (!row || Number(row.dataset.bookIndex) === focus) return
    select(Number(row.dataset.bookIndex))
    focusCamera(focus)
  }
  indexPane.addEventListener('pointermove', selectFromIndex, { signal })
  indexPane.addEventListener(
    'focusin',
    event => {
      followIndexScroll = false
      selectFromIndex(event)
    },
    { signal },
  )
  const enableIndexScroll = () => {
    followIndexScroll = true
  }
  indexPane.addEventListener('wheel', enableIndexScroll, { passive: true, signal })
  indexPane.addEventListener('pointerdown', enableIndexScroll, { signal })
  indexPane.addEventListener(
    'keydown',
    event => {
      if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key))
        enableIndexScroll()
    },
    { signal },
  )
  indexPane.addEventListener(
    'scroll',
    () => {
      if (!followIndexScroll || dialog.open || indexFrame) return
      indexFrame = requestAnimationFrame(() => {
        indexFrame = 0
        if (signal.aborted || dialog.open) return
        const visible = matches().filter(row => row.offsetParent !== null)
        if (!visible.length) return
        const pane = indexPane.getBoundingClientRect()
        const line = pane.top + pane.height * 0.4
        let nearest = visible[0]
        let distance = Infinity
        for (const row of visible) {
          const bounds = row.getBoundingClientRect()
          const delta = Math.abs(bounds.top + bounds.height / 2 - line)
          if (delta < distance) {
            nearest = row
            distance = delta
          }
        }
        if (indexPane.scrollTop <= 0) nearest = visible[0]
        else if (indexPane.scrollTop + indexPane.clientHeight >= indexPane.scrollHeight - 1)
          nearest = visible[visible.length - 1]
        const index = Number(nearest.dataset.bookIndex)
        if (index === focus) return
        select(index)
        focusCamera(index)
      })
    },
    { passive: true, signal },
  )
  const disposeEscape = registerEscapeHandler(dialog, () => dialog.close())
  dialog.addEventListener(
    'click',
    event => {
      if (event.target !== dialog) return
      const bounds = dialog.getBoundingClientRect()
      if (
        event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom
      )
        dialog.close()
    },
    { signal },
  )
  dialog.addEventListener(
    'close',
    () => {
      hardcover?.clear()
      const fallback = entryByIndex
        .get(focus)
        ?.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
      const target = returnFocus?.isConnected ? returnFocus : fallback
      target?.focus({ preventScroll: true })
    },
    { signal },
  )
  dialog.addEventListener(
    'wheel',
    event => {
      if (event.ctrlKey || event.metaKey || !dialog.open) return
      const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
      if (!delta) return
      const body =
        event.target instanceof Element ? event.target.closest('.bookcase-preview-body') : null
      if (
        body instanceof HTMLElement &&
        ((delta > 0 && body.scrollTop + body.clientHeight < body.scrollHeight - 1) ||
          (delta < 0 && body.scrollTop > 0))
      )
        return
      event.preventDefault()
      const now = performance.now()
      if (now - previewWheelStep < 220) return
      if (now - previewWheelAt > 160 || Math.sign(delta) !== Math.sign(previewWheelDelta))
        previewWheelDelta = 0
      previewWheelAt = now
      previewWheelDelta += delta * wheelUnit(event, dialog.clientHeight)
      if (Math.abs(previewWheelDelta) < 64) return
      browsePreview(Math.sign(previewWheelDelta))
      previewWheelDelta = 0
      previewWheelStep = now
    },
    { passive: false, signal },
  )
  dialog.addEventListener(
    'keydown',
    event => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault()
        browsePreview(event.key === 'ArrowLeft' ? -1 : 1)
      }
    },
    { signal },
  )

  const refresh = () => {
    const searching = Boolean(
      document.querySelector<HTMLInputElement>('#base-search-input')?.value.trim(),
    )
    if (searching && !searchOpen) searchOpen = new Map(groups.map(group => [group, group.open]))
    for (const group of groups) {
      const shown = [...group.querySelectorAll<HTMLLIElement>('.bookcase-entry')].filter(
        entry => entry.style.display !== 'none' && !entry.hasAttribute('data-topic-hidden'),
      )
      const count = group.querySelector('.bookcase-index-count')
      if (count) count.textContent = String(shown.length)
      if (searching && shown.length) group.open = true
      else if (!searching && searchOpen) group.open = searchOpen.get(group) ?? true
    }
    if (!searching) searchOpen = undefined
    const visible = matches()
    const rearranged = arrange(visible)
    // A new topic or search keeps the chosen book only while it stays in view.
    const selected = visible.find(entry => Number(entry.dataset.bookIndex) === focus)
    root.toggleAttribute('data-empty', visible.length === 0)
    if (selected) {
      select(focus)
      if (rearranged) focusCamera(focus)
    } else deselect()
    if (dialog.open) {
      if (selected) renderPreview()
      else dialog.close()
    }
  }
  document.addEventListener('base-search-results', refresh, { signal })

  const topicButtons = [...root.querySelectorAll<HTMLButtonElement>('.bookcase-topic')]
  const showTopic = (next: string) => {
    topic = topicButtons.some(button => button.dataset.topic === next) ? next : ''
    for (const button of topicButtons) {
      button.setAttribute('aria-selected', String(button.dataset.topic === topic))
      button.tabIndex = button.dataset.topic === topic ? 0 : -1
    }
    for (const entry of entries) {
      const topics: string[] = JSON.parse(entry.dataset.topics ?? '[]')
      entry.toggleAttribute('data-topic-hidden', Boolean(topic) && !topics.includes(topic))
    }
    for (const group of groups) {
      const empty = !group.querySelector('.bookcase-entry:not([data-topic-hidden])')
      group.toggleAttribute('data-topic-hidden', empty)
    }
    const url = new URL(window.location.href)
    if (topic) url.searchParams.set('topic', topic)
    else url.searchParams.delete('topic')
    history.replaceState(history.state, '', url)
    refresh()
  }
  const topicList = root.querySelector<HTMLElement>('.bookcase-topics')
  viewport.id ||= `bookcase-viewport-${crypto.randomUUID()}`
  for (const button of topicButtons) {
    button.setAttribute('aria-controls', viewport.id)
    button.addEventListener('click', () => showTopic(button.dataset.topic ?? ''), { signal })
  }
  const nextTab = (key: string, active: number): number | undefined => {
    const n = topicButtons.length
    if (key === 'ArrowUp' || key === 'ArrowLeft') return (active - 1 + n) % n
    if (key === 'ArrowDown' || key === 'ArrowRight') return (active + 1) % n
    if (key === 'Home') return 0
    if (key === 'End') return n - 1
    // A letter moves to the next tab whose shortcut starts with it, as the triathlon tabs do.
    if (key.length !== 1) return undefined
    for (let step = 1; step <= n; step++) {
      const i = (active + step) % n
      if (topicButtons[i].dataset.shortcut === key.toLowerCase()) return i
    }
    return undefined
  }
  topicList?.addEventListener(
    'keydown',
    event => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return
      const active = topicButtons.findIndex(button => button.dataset.topic === topic)
      const next = nextTab(event.key, Math.max(0, active))
      if (next === undefined) return
      event.preventDefault()
      showTopic(topicButtons[next].dataset.topic ?? '')
      topicButtons[next].focus()
    },
    { signal },
  )

  const resize = () => {
    for (const view of aim ? [camera, aim] : [camera]) {
      view.x += (viewport.clientWidth - viewportWidth) / 2
      view.y += (viewport.clientHeight - viewportHeight) / 2
    }
    viewportWidth = viewport.clientWidth
    viewportHeight = viewport.clientHeight
    const previousFit = fitScale
    fitScale = Math.max(
      0.01,
      Math.min((viewport.clientWidth - 32) / box.width, (viewport.clientHeight - 72) / box.height),
    )
    if (fitted || camera.scale < fitScale || previousFit === 1) fit()
    else schedule()
  }
  const observer = new ResizeObserver(resize)
  observer.observe(root)
  observer.observe(viewport)
  window.addEventListener('resize', resize, { signal })
  resize()
  const initialTopic = new URL(window.location.href).searchParams.get('topic')
  if (initialTopic) {
    // A shared link opens on its tab without the label animation.
    if (topicList) topicList.dataset.motion = 'instant'
    showTopic(initialTopic)
    topicList?.getBoundingClientRect()
    if (topicList) delete topicList.dataset.motion
  }
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      disposeEscape()
      if (frame) cancelAnimationFrame(frame)
      if (indexFrame) cancelAnimationFrame(indexFrame)
      if (dialog.open) dialog.close()
      dropLift()
      for (const image of residentCovers.values()) removeArtwork(image)
      residentCovers.clear()
      preloaded.clear()
      delete root.dataset.bookcaseReady
    },
    { once: true },
  )
}

document.addEventListener('nav', () => {
  const covers = new Map<string, Cover>()
  const initializeBookcases = () => {
    for (const root of document.querySelectorAll<HTMLElement>('.bookcase'))
      initializeBookcase(root, covers)
  }
  initializeBookcases()
  document.addEventListener('base-content-changed', initializeBookcases)
  window.addCleanup(() => {
    document.removeEventListener('base-content-changed', initializeBookcases)
    covers.clear()
  })
})

export default ''
