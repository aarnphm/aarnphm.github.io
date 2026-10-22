import { toHtml } from 'hast-util-to-html'
import type { BookcaseScene } from './scene'
import {
  BOOKCASE_DEFAULTS,
  packShelves,
  type ShelfGroup,
} from '../../../util/base/bookcase/geometry'
import { bookcaseScene, type SceneBook } from '../../../util/base/bookcase/scene'
import {
  BOOKCASE_STYLES,
  placeShelves,
  SHELF_MATERIALS,
  stylePackOptions,
} from '../../../util/base/bookcase/styles'
import { rootNavSignal } from '../root-lifecycle'
import { removeArtwork } from './artwork'
import { retainCovers } from './covers'
import { readBookBounds } from './state'

export function mountShelves(s: BookcaseScene) {
  s.scrollSpace = document.createElement('div')

  s.scrollSpace.className = 'bookcase-scroll-space'

  s.scrollSpace.setAttribute('aria-hidden', 'true')

  s.pan = document.createElement('div')

  s.pan.className = 'bookcase-pan'

  s.world.before(s.scrollSpace, s.pan)

  s.pan.append(s.world)

  s.viewportWidth = s.viewport.clientWidth

  s.viewportHeight = s.viewport.clientHeight

  retainCovers(s.root, s.covers)

  s.root.dataset.bookcaseReady = ''

  s.signal = rootNavSignal(s.root)

  s.loadTextures = () => {
    for (const image of s.scene.querySelectorAll<SVGImageElement>('image[data-bookcase-texture]')) {
      const url = image.dataset.bookcaseTexture
      if (url) image.setAttribute('href', url)
    }
  }

  s.loadTextures()

  s.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

  s.entries = [...s.root.querySelectorAll<HTMLLIElement>('.bookcase-entry')]

  s.groups = [...s.root.querySelectorAll<HTMLDetailsElement>('.bookcase-index-group')]

  s.books = new Map<number, SVGGElement>()

  s.bookBounds = new Map<number, DOMRect>()

  s.spineLengths = new Map<number, number>()

  s.spineBounds = new Map<number, { x: number; y: number; width: number; height: number }>()

  s.cacheBookGeometry = (index: number, book: SVGGElement) => {
    s.bookBounds.set(index, readBookBounds(book))
    const transform = book.dataset.spineTransform
    const [width, height] = (book.dataset.spineSize ?? '').split(' ').map(Number)
    if (!transform || !Number.isFinite(width) || !Number.isFinite(height)) return
    const values = transform.slice(7, -1).trim().split(/[ ,]+/).map(Number)
    if (values.length !== 6 || !values.every(Number.isFinite)) return
    const [a, b, c, d, e, f] = values
    s.spineLengths.set(index, Math.hypot(c, d) * height)
    const xs = [0, a * width, c * height, a * width + c * height]
    const ys = [0, b * width, d * height, b * width + d * height]
    s.spineBounds.set(index, {
      x: e + Math.min(...xs),
      y: f + Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
    })
  }

  s.entryByIndex = new Map(s.entries.map(entry => [Number(entry.dataset.bookIndex), entry]))

  for (const book of s.scene.querySelectorAll<SVGGElement>('.bc-book'))
    s.books.set(Number(book.dataset.bookIndex), book)

  s.residentCovers = new Map<string, SVGImageElement>()

  // Base art the camera is heading toward, fetched and decoded into detached images so a new

  // Base art the camera is heading toward, fetched and decoded into detached images so a new
  // resident node finds its URL in the document's image cache. Insertion order is recency.

  // Base art the camera is heading toward, fetched and decoded into detached images so a new
  // resident node finds its URL in the document's image cache. Insertion order is recency.
  s.prefetched = new Map<string, { image: HTMLImageElement; settled: boolean }>()

  s.prefetching = 0

  s.idle = typeof requestIdleCallback === 'function'

  s.coverWindow = ''

  ;[s.originX, s.originY] = (s.scene.dataset.origin ?? '0 0').split(' ').map(Number)

  s.box = s.scene.viewBox.baseVal

  s.world.style.width = `${s.box.width}px`

  s.world.style.height = `${s.box.height}px`

  for (const [index, book] of s.books) s.cacheBookGeometry(index, book)

  s.inventory = new Map(
    [...s.books].map(([index, book]) => [
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
}

export function mountArrangement(s: BookcaseScene) {
  // The chosen topic (a group, a tag, "other", or "" for all) lives in the page URL as ?topic=.

  // The chosen topic (a group, a tag, "other", or "" for all) lives in the page URL as ?topic=.
  s.topic = ''

  s.topicShelves = JSON.parse(s.root.dataset.topicShelves ?? '{}')

  s.arrangement = `|${s.entries.map(entry => entry.dataset.bookIndex).join(',')}`

  s.arrange = (visible: HTMLLIElement[]) => {
    const key = `${s.topic}|${visible.map(entry => entry.dataset.bookIndex).join(',')}`
    if (key === s.arrangement) return false
    s.arrangement = key
    const sections = new Map<Element | string | null, ShelfGroup<SceneBook & { index: number }>>()
    for (const entry of visible) {
      const index = Number(entry.dataset.bookIndex)
      const book = s.inventory.get(index)
      if (!book) continue
      // Inside a topic, shelves group by the topic's own property instead of the index groups.
      const section = entry.closest('.bookcase-index-group')
      const slot = s.topic ? (entry.dataset.topicShelf ?? null) : section
      let group = sections.get(slot)
      if (!group) {
        const label = s.topic
          ? entry.dataset.topicShelf
          : section?.querySelector('.bookcase-index-label')?.textContent
        group = { label: label ?? null, books: [] }
        sections.set(slot, group)
      }
      group.books.push(book)
    }
    const shelves = [...sections.values()]
    if (s.topic)
      shelves.sort((a, b) =>
        a.label === null ? 1 : b.label === null ? -1 : a.label.localeCompare(b.label),
      )
    const own = s.topic ? s.topicShelves[s.topic] : undefined
    const style =
      own?.style ?? BOOKCASE_STYLES.find(name => name === s.root.dataset.shelfStyle) ?? 'straight'
    const material = own
      ? own.material
      : SHELF_MATERIALS.find(name => name === s.root.dataset.shelfMaterial)
    const rows = Number(s.root.dataset.shelfRows) || BOOKCASE_DEFAULTS.shelfRows
    const options = {
      bookHeight: Number(s.root.dataset.bookHeight) || BOOKCASE_DEFAULTS.bookHeight,
      shelfWidth: Number(s.root.dataset.shelfWidth) || BOOKCASE_DEFAULTS.shelfWidth,
      // A topic's shelves group by small subcategories, so they share runs.
      continuous: Boolean(s.topic),
    }
    const layout = placeShelves(
      packShelves(shelves, stylePackOptions(style, options, visible.length, rows)),
      style,
      material,
      { showTags: own?.showTags ?? s.root.dataset.showTags !== 'false', rows },
    )
    for (const book of layout.books) book.index = book.item.index
    const name = s.root.closest<HTMLElement>('.base-view')?.dataset.baseViewName ?? 'bookcase'
    const template = document.createElement('template')
    template.innerHTML = toHtml(
      bookcaseScene(
        layout,
        `${name}: ${layout.books.length} books on ${layout.shelves.length} shelves`,
      ),
    )
    const next = template.content.querySelector('svg')
    if (!next) return false
    s.dropLift()
    for (const image of s.residentCovers.values()) removeArtwork(image)
    for (const attribute of next.attributes) s.scene.setAttribute(attribute.name, attribute.value)
    s.scene.replaceChildren(...next.childNodes)
    s.loadTextures()
    ;[s.originX, s.originY] = (s.scene.dataset.origin ?? '0 0').split(' ').map(Number)
    s.box = s.scene.viewBox.baseVal
    s.fitWidth = Number(s.scene.dataset.fit) || s.box.width
    s.bakedScale = 1
    s.scene.style.width = `${s.box.width}px`
    s.scene.style.height = `${s.box.height}px`
    s.world.style.width = `${s.box.width}px`
    s.world.style.height = `${s.box.height}px`
    s.world.style.transform = ''
    s.world.style.willChange = 'auto'
    s.scrollLayout = ''
    s.books.clear()
    s.bookBounds.clear()
    s.spineLengths.clear()
    s.spineBounds.clear()
    s.residentCovers.clear()
    s.coverWindow = ''
    for (const book of s.scene.querySelectorAll<SVGGElement>('.bc-book')) {
      const index = Number(book.dataset.bookIndex)
      const fullCover = s.inventory.get(index)?.fullCover
      if (fullCover) book.dataset.fullCover = fullCover
      s.books.set(index, book)
      s.cacheBookGeometry(index, book)
    }
    s.fitted = true
    s.updateFit()
    return true
  }
}
