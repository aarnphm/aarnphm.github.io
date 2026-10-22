import type { BookcaseScene } from './scene'

export const SVG_NS = 'http://www.w3.org/2000/svg'

const MAX_SHELF_COVERS = 32
// Spines get their own slots: a shelf in view shows far more spines than jackets, and a spine
// texture decodes to about a tenth of a cover.
const MAX_SHELF_SPINES = 64
const MAX_DETAIL_SPINES = 4
const DETAIL_ENTER_PX = 360
const DETAIL_LEAVE_PX = 280
const COVER_MARGIN = 120
// Resident art stays until its book is this far out, so dragging back and forth keeps it.
const EVICT_MARGIN = 360
// Prefetch reaches where the camera will be after this much travel at its current velocity.
const PREFETCH_LOOKAHEAD_MS = 400
const PREFETCH_MARGIN = 240
const MAX_PREFETCH_INFLIGHT = 4
const MAX_PREFETCHED = 48

type ArtworkFace = 'cover' | 'spine'

// Each resident face has one SVG image. A temporary decoder keeps its low texture visible
// until the detail is ready; eviction or a new tier request cancels that decoder.
const spineDecoders = new WeakMap<SVGImageElement, HTMLImageElement>()
export function cancelSpineDetail(image: SVGImageElement): void {
  delete image.dataset.spineResolutionPending
  const decoder = spineDecoders.get(image)
  if (!decoder) return
  spineDecoders.delete(image)
  decoder.removeAttribute('src')
}

export function setSpineResolution(image: SVGImageElement, detail: boolean): void {
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

export function removeArtwork(image: SVGImageElement): void {
  if (image.classList.contains('bc-spine-art')) {
    const book = image.closest<SVGGElement>('.bc-book')
    if (book) delete book.dataset.spineArtReady
    cancelSpineDetail(image)
  }
  image.remove()
}

export function addArtwork(
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

export const addCover = (book: SVGGElement, selected = false) => addArtwork(book, 'cover', selected)

// The camera owns resident images; a lift may also reveal a temporary neighbour cover.
export function removeExposedCover(image: SVGImageElement | undefined): void {
  if (image && !image.hasAttribute('data-shelf-resident')) image.remove()
}

export function installCovers(s: BookcaseScene) {
  s.refreshShelfCovers = function refreshShelfCovers(): void {
    const pixelRatio = Math.min(2, window.devicePixelRatio || 1)
    // A moving camera keeps detail spines it has but starts no new detail decodes.
    const moving = Boolean(s.drag?.moved || s.aim || s.wheelTimer !== undefined)
    const key = `${s.focus}:${Math.round(s.camera.x / 64)}:${Math.round(s.camera.y / 64)}:${Math.round(s.camera.scale * 32)}:${s.viewportWidth}:${s.viewportHeight}:${pixelRatio}:${moving}`
    if (key === s.coverWindow) return
    s.coverWindow = key
    const candidates: {
      index: number
      face: ArtworkFace
      key: string
      score: number
      exposed: boolean
      outside: boolean
      detail?: boolean
    }[] = []
    for (const [index, book] of s.books) {
      const bounds = s.bookBounds.get(index)
      if (!bounds || book.hasAttribute('data-lifted')) continue
      const width = bounds.width * s.camera.scale
      const height = bounds.height * s.camera.scale
      const x = (bounds.x - s.originX) * s.camera.scale + s.camera.x
      const y = (bounds.y - s.originY) * s.camera.scale + s.camera.y
      const gap = Math.max(-x - width, x - s.viewportWidth, -y - height, y - s.viewportHeight)
      if (height < 72 || gap > EVICT_MARGIN) continue
      // Past the add margin a book keeps the art it has but gets no new art.
      const outside = gap > COVER_MARGIN
      const end = !book.nextElementSibling?.matches('.bc-book:not([data-lifted])')
      const distance =
        Math.abs(y + height / 2 - s.viewportHeight / 2) +
        Math.abs(x + width / 2 - s.viewportWidth / 2) * 0.25
      if (
        book.dataset.coverTransform &&
        book.dataset.cover &&
        (!outside || s.residentCovers.has(`${index}:cover`))
      )
        candidates.push({
          index,
          face: 'cover',
          key: `${index}:cover`,
          score: distance,
          exposed: end,
          outside,
        })
      const key = `${index}:spine`
      const resident = s.residentCovers.get(key)
      if (book.dataset.spineArt && (!outside || resident)) {
        const detailed =
          resident?.dataset.spineResolution === 'detail' ||
          Boolean(resident && spineDecoders.has(resident))
        const length = (s.spineLengths.get(index) ?? 0) * s.camera.scale * pixelRatio
        const spine = s.spineBounds.get(index)
        const sx = spine ? (spine.x - s.originX) * s.camera.scale + s.camera.x : -Infinity
        const sy = spine ? (spine.y - s.originY) * s.camera.scale + s.camera.y : -Infinity
        const visible =
          spine &&
          sx + spine.width * s.camera.scale >= 0 &&
          sx <= s.viewportWidth &&
          sy + spine.height * s.camera.scale >= 0 &&
          sy <= s.viewportHeight
        candidates.push({
          index,
          face: 'spine',
          key,
          score: distance * 0.5,
          exposed: false,
          outside,
          detail: Boolean(
            visible &&
            (detailed || !moving) &&
            book.dataset.spineArtDetail &&
            book.dataset.spineArtDetailFailed !== book.dataset.spineArtDetail &&
            length > (detailed ? DETAIL_LEAVE_PX : DETAIL_ENTER_PX),
          ),
        })
      }
    }
    // Exposed jackets need a slot even at the top or bottom of the fitted case; art kept past
    // the add margin only fills slots nothing nearer wants.
    const slots = { cover: MAX_SHELF_COVERS, spine: MAX_SHELF_SPINES }
    const wanted = candidates
      .sort(
        (a, b) =>
          Number(b.exposed) - Number(a.exposed) ||
          Number(a.outside) - Number(b.outside) ||
          a.score - b.score,
      )
      .filter(candidate => slots[candidate.face]-- > 0)
    const keys = new Set(wanted.map(candidate => candidate.key))
    for (const [key, image] of s.residentCovers) {
      if (keys.has(key) && image.isConnected) continue
      removeArtwork(image)
      s.residentCovers.delete(key)
    }
    // Downgrade outgoing detail slots before decoding replacements, including pending loads.
    const detailed = new Set(
      wanted
        .filter(candidate => candidate.detail)
        .slice(0, MAX_DETAIL_SPINES)
        .map(candidate => candidate.key),
    )
    for (const [key, image] of s.residentCovers) {
      if (image.classList.contains('bc-spine-art') && !detailed.has(key))
        setSpineResolution(image, false)
    }
    for (const { index, face, key } of wanted) {
      const book = s.books.get(index)
      let image = s.residentCovers.get(key)
      if (!image && book) {
        image = addArtwork(book, face)
        if (image) {
          image.dataset.shelfResident = ''
          s.residentCovers.set(key, image)
        }
      }
      if (image && face === 'spine') setSpineResolution(image, detailed.has(key))
    }
  }

  // Ranks the base art (never the detail tier) of books in the views the camera is heading
  // toward: its position after the lookahead at the current velocity, and a glide's target.
  s.planPrefetch = function planPrefetch(): void {
    s.prefetchTask = undefined
    if (s.signal.aborted || s.dialog?.open) return
    const recent = performance.now() - s.painted.at < 150
    const dx = recent
      ? s.clamp(s.velocity.x * PREFETCH_LOOKAHEAD_MS, -s.viewportWidth, s.viewportWidth)
      : 0
    const dy = recent
      ? s.clamp(s.velocity.y * PREFETCH_LOOKAHEAD_MS, -s.viewportHeight, s.viewportHeight)
      : 0
    const views = [{ camera: s.camera, dx, dy }]
    if (s.aim) views.push({ camera: s.aim, dx: 0, dy: 0 })
    const ranked: { url: string; score: number }[] = []
    for (const [index, book] of s.books) {
      const bounds = s.bookBounds.get(index)
      if (!bounds) continue
      let score = Infinity
      for (const view of views) {
        const { scale } = view.camera
        const width = bounds.width * scale
        const height = bounds.height * scale
        const x = (bounds.x - s.originX) * scale + view.camera.x
        const y = (bounds.y - s.originY) * scale + view.camera.y
        if (height < 72) continue
        // The swept window between now and the projected view, measured in this view's pixels.
        const gap = Math.max(
          Math.min(0, -view.dx) - x - width,
          x - Math.max(s.viewportWidth, s.viewportWidth - view.dx),
          Math.min(0, -view.dy) - y - height,
          y - Math.max(s.viewportHeight, s.viewportHeight - view.dy),
        )
        if (gap > PREFETCH_MARGIN) continue
        // Nearest the current viewport first: that art is needed soonest.
        score = Math.min(
          score,
          Math.max(0, -x - width, x - s.viewportWidth) +
            Math.max(0, -y - height, y - s.viewportHeight),
        )
      }
      if (score === Infinity) continue
      const spine = book.dataset.spineArt
      if (
        spine?.includes('/static/bookcase-spines/') &&
        book.dataset.spineArtFailed !== spine &&
        !s.residentCovers.has(`${index}:spine`)
      )
        ranked.push({ url: spine, score: score * 0.5 })
      const cover = book.dataset.cover
      if (
        book.dataset.coverTransform &&
        cover?.includes('/static/bookcase/') &&
        book.dataset.coverFailed !== cover &&
        !s.residentCovers.has(`${index}:cover`)
      )
        ranked.push({ url: cover, score })
    }
    const wanted = ranked
      .sort((a, b) => a.score - b.score)
      .slice(0, MAX_PREFETCHED)
      .map(entry => entry.url)
    // A resident node shares the load and its decode, so its prefetch is kept.
    const keep = new Set(wanted)
    for (const image of s.residentCovers.values()) keep.add(image.getAttribute('href') ?? '')
    for (const [url, entry] of s.prefetched)
      if (!entry.settled && !keep.has(url)) s.dropPrefetch(url)
    // Refresh recency from least to most wanted, so eviction drops the least wanted first.
    for (const url of [...wanted].reverse()) {
      const entry = s.prefetched.get(url)
      if (!entry) continue
      s.prefetched.delete(url)
      s.prefetched.set(url, entry)
    }
    for (const url of wanted) {
      if (s.prefetching >= MAX_PREFETCH_INFLIGHT) break
      if (s.prefetched.has(url)) continue
      const image = new Image()
      image.decoding = 'async'
      image.fetchPriority = 'low'
      s.prefetched.set(url, { image, settled: false })
      s.prefetching++
      image.src = url
      const settle = () => {
        if (s.prefetched.get(url)?.image !== image) return
        s.settlePrefetch(url)
        s.schedulePrefetch()
      }
      void image.decode().then(settle, settle)
    }
    for (const url of s.prefetched.keys()) {
      if (s.prefetched.size <= MAX_PREFETCHED) break
      s.dropPrefetch(url)
    }
  }

  s.schedulePrefetch = function schedulePrefetch(): void {
    if (s.prefetchTask !== undefined || s.signal.aborted) return
    s.prefetchTask = s.idle
      ? requestIdleCallback(s.planPrefetch, { timeout: 100 })
      : window.setTimeout(s.planPrefetch, 50)
  }
}

export function mountCovers(s: BookcaseScene) {
  s.schedule = () => {
    if (!s.frame) s.frame = requestAnimationFrame(s.paint)
  }

  s.settlePrefetch = (url: string) => {
    const entry = s.prefetched.get(url)
    if (!entry || entry.settled) return
    entry.settled = true
    s.prefetching--
  }

  s.dropPrefetch = (url: string) => {
    const entry = s.prefetched.get(url)
    if (!entry) return
    s.settlePrefetch(url)
    s.prefetched.delete(url)
    // Clearing the source aborts a load that no resident node shares.
    entry.image.removeAttribute('src')
  }

  // Ranks the base art (never the detail tier) of books in the views the camera is heading

  // Ranks the base art (never the detail tier) of books in the views the camera is heading
  // toward: its position after the lookahead at the current velocity, and a glide's target.
}
