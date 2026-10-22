import {
  BOOKCASE_DEFAULTS,
  COS30,
  type BookcaseLayout,
  type PackOptions,
  type Point,
  type Segment,
} from './geometry'
import { corner } from './styles/corner-style'
import { ladder } from './styles/ladder-style'
import { spine } from './styles/spine-style'
import { straight } from './styles/straight-style'
import {
  BOOKCASE_STYLES,
  SHELF_MATERIALS,
  WALL,
  type BookcaseStyle,
  type Built,
  type ShelfMaterial,
  type ShelfPlacement,
  type StyleContext,
  type StyledLayout,
  type StyleHalo,
  type StyleLayer,
  type StylePart,
  type WorldBook,
} from './styles/style-model'
import { wallHung } from './styles/wall-style'
import { bookFacesW, pullOffsetW, worldBook } from './styles/world-space'
import { zigzag } from './styles/zigzag-style'
export {
  BOOKCASE_STYLES,
  SHELF_MATERIALS,
  type ShelfMaterial,
  type StyleHalo,
  type StyleSolid,
  type StyledLayout,
  type WorldBook,
} from './styles/style-model'
export { bookDetailW } from './styles/world-space'
export {
  bookFacesW,
  coverMatrixW,
  pullOffsetW,
  sheenFill,
  spineCentreW,
  spineMatrixW,
  spineTextW,
} from './styles/world-space'
export { faceFill, grainFill, materialCss, materialDefs } from './styles/material-paint'
export type { DefNode } from './styles/grain-recipes'

/**
 * A topic is a value of the view's groupBy property or a `topicShelves` key. A book belongs to a
 * topic when its groupBy value is the topic or when it carries the topic as a tag.
 */
export interface TopicShelf {
  style: BookcaseStyle
  material?: ShelfMaterial
  /** Overrides the view's `showTags` for this topic. */
  showTags?: boolean
}

/** Used when the view has no `topicShelves` entry for a topic. */
export const TOPIC_SHELVES: Readonly<Record<string, TopicShelf>> = {
  philosophy: { style: 'wall', material: 'walnut' },
  'p/fiction': { style: 'corner', material: 'paper' },
  'p/nf': { style: 'ladder', material: 'birch-ply' },
  'p/scifi': { style: 'spine', material: 'copper' },
}

const TOPIC_WORDS: Readonly<Record<string, string>> = { nf: 'non-fiction', scifi: 'sci-fi' }

const entryOf = (shelves: unknown, topic: string): Record<string, unknown> | undefined => {
  const entry =
    shelves && typeof shelves === 'object' ? (shelves as Record<string, unknown>)[topic] : null
  return entry && typeof entry === 'object' ? (entry as Record<string, unknown>) : undefined
}

/** Keys of the view's `topicShelves`. */
export function topicKeys(shelves?: unknown): string[] {
  return shelves && typeof shelves === 'object' ? Object.keys(shelves) : []
}

/** The entry's `displayNames`, else the value without its `p/` prefix. */
export function topicLabel(topic: string, shelves?: unknown): string {
  const own = entryOf(shelves, topic)?.displayNames
  if (typeof own === 'string' && own.trim()) return own.trim()
  const bare = topic.replace(/^p\//, '')
  return TOPIC_WORDS[bare] ?? bare
}

/**
 * The view's `topicShelves` entry when its style is known, else the default for the topic. The
 * entry's `showTags` applies either way.
 */
export function topicShelf(topic: string, shelves?: unknown): TopicShelf | undefined {
  const own = entryOf(shelves, topic)
  const style = BOOKCASE_STYLES.find(name => name === own?.style)
  const shelf: TopicShelf | undefined = style
    ? { style, material: SHELF_MATERIALS.find(name => name === own?.material) }
    : TOPIC_SHELVES[topic]
  return shelf && typeof own?.showTags === 'boolean' ? { ...shelf, showTags: own.showTags } : shelf
}

/** Shortest unique uppercase prefix of each label, shown on a collapsed topic tab. */
export function topicShortcuts(labels: readonly string[]): string[] {
  const words = labels.map(label => label.replace(/[^\p{L}\p{N}]/gu, '').toUpperCase() || '?')
  return words.map((word, i) => {
    for (let n = 1; n < word.length; n++) {
      const head = word.slice(0, n)
      if (words.every((other, j) => j === i || !other.startsWith(head))) return head
    }
    return word
  })
}

/** True when a tag list holds the topic itself or a tag nested under it (`p/nf/essays`). */
export function hasTopicTag(tags: unknown, topic: string): boolean {
  if (!Array.isArray(tags)) return false
  return tags.some(tag => {
    if (typeof tag !== 'string') return false
    const bare = tag.replace(/^#/, '')
    return bare === topic || bare.startsWith(`${topic}/`)
  })
}

/**
 * Run length for a style. The spine style packs a stack height and the wall style short blades;
 * the ladder needs longer boards so it stays within one bay of tiers. These bypass BOOKCASE_LIMITS, which
 * only clamp user input.
 */
export function stylePackOptions(
  style: BookcaseStyle,
  options: PackOptions,
  bookCount = 0,
  rows: number = BOOKCASE_DEFAULTS.shelfRows,
): PackOptions {
  switch (style) {
    case 'spine':
      return { ...options, shelfWidth: 160 }
    case 'wall':
      return { ...options, shelfWidth: 320 }
    case 'ladder':
      return {
        ...options,
        shelfWidth: Math.max(
          options.shelfWidth ?? BOOKCASE_DEFAULTS.shelfWidth,
          Math.ceil((bookCount * 26.5) / rows),
        ),
      }
    default:
      return options
  }
}

const STYLE_BUILDERS: Record<BookcaseStyle, (ctx: StyleContext) => Built> = {
  straight,
  corner,
  spine,
  wall: wallHung,
  zigzag,
  ladder,
}

/** Each style's own materials: a base for every part, then per-part exceptions. */
const STYLE_MATERIALS: Record<
  BookcaseStyle,
  [ShelfMaterial, Partial<Record<StylePart, ShelfMaterial>>]
> = {
  straight: ['paper', {}],
  corner: ['walnut', {}],
  spine: ['copper', {}],
  wall: ['birch-ply', {}],
  zigzag: ['black-lacquer', { upright: 'walnut', foot: 'walnut' }],
  ladder: ['birch-ply', { rail: 'steel', stretcher: 'steel' }],
}

/**
 * Parts a material brings with it, after the style's own exceptions: a concrete case carries black
 * boards, a copper case sits on a black plinth.
 */
const COMPANIONS: Partial<Record<ShelfMaterial, Partial<Record<StylePart, ShelfMaterial>>>> = {
  concrete: { board: 'black-lacquer' },
  copper: { plinth: 'black-lacquer', foot: 'black-lacquer', base: 'black-lacquer' },
}

export interface PlaceOptions {
  /** False drops every shelf's label plate (the `showTags` view key). */
  showTags?: boolean
  /** Shelf levels in one bay (the `shelfRows` view key). */
  rows?: number
}

/** Runs that share one level: the corner's two wings, the wall's two columns. */
const RUNS_PER_LEVEL: Partial<Record<BookcaseStyle, number>> = { corner: 2, wall: 2 }
/** Screen gap between two bays. */
const BAY_GAP = 48

function screenBounds(
  layers: StyleLayer[],
  worldBooks: WorldBook[],
): readonly [number, number, number, number] {
  const corners: Point[] = []
  for (const layer of layers) {
    if (layer.kind === 'solids') {
      for (const solid of layer.solids) for (const face of solid.faces) corners.push(...face.points)
      for (const h of layer.halos) corners.push(...h.points)
    } else {
      corners.push(...layer.contact.points, ...(layer.plate?.face ?? []))
    }
  }
  for (const b of worldBooks) {
    const [dx, dy] = pullOffsetW(b)
    for (const [x, y] of bookFacesW(b).edge) corners.push([x, y], [x + dx, y + dy])
  }
  const xs = corners.map(p => p[0])
  const ys = corners.map(p => p[1])
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
}

/** Moves a layer along the screen's x. */
function shiftLayer(layer: StyleLayer, dx: number): StyleLayer {
  const pt = ([x, y]: Point): Point => [x + dx, y]
  const sg = ([a, b]: Segment): Segment => [pt(a), pt(b)]
  const moveHalo = (h: StyleHalo): StyleHalo => ({ ...h, points: h.points.map(pt) })
  if (layer.kind === 'shelf') {
    const plate = layer.plate
    if (!plate) return { ...layer, contact: moveHalo(layer.contact) }
    const [a, b, c, d, e, f] = plate.matrix
    return {
      ...layer,
      contact: moveHalo(layer.contact),
      plate: {
        ...plate,
        face: plate.face.map(pt),
        ticks: plate.ticks.map(sg),
        matrix: [a, b, c, d, e + dx, f],
      },
    }
  }
  return {
    ...layer,
    halos: layer.halos.map(moveHalo),
    solids: layer.solids.map(solid => ({
      ...solid,
      faces: solid.faces.map(face => ({ ...face, points: face.points.map(pt) })),
      lines: solid.lines.map(line => ({ ...line, segments: line.segments.map(sg) })),
      creases: solid.creases.map(sg),
      outline: solid.outline?.map(pt) ?? null,
      edges: solid.edges.map(sg),
      screws: solid.screws.map(pt),
    })),
  }
}

/**
 * Places each packed run for a style and lists every part in painter's order. Pure. `materials`
 * replaces the style's main material and keeps its per-part exceptions (SPEC 15.3), or gives
 * per-part overrides.
 */
export function placeShelves<T>(
  layout: BookcaseLayout<T>,
  style: BookcaseStyle,
  materials?: ShelfMaterial | Partial<Record<StylePart, ShelfMaterial>>,
  options: PlaceOptions = {},
): StyledLayout<T> {
  const [main, own] = STYLE_MATERIALS[style]
  const base = typeof materials === 'string' ? materials : main
  const parts = typeof materials === 'string' ? undefined : materials
  const pick = (part: StylePart): ShelfMaterial =>
    parts?.[part] ?? own[part] ?? COMPANIONS[base]?.[part] ?? base
  const showTags = options.showTags ?? true
  const rows = Math.max(1, Math.round(options.rows ?? BOOKCASE_DEFAULTS.shelfRows))
  const perBay = rows * (RUNS_PER_LEVEL[style] ?? 1)
  const placements: ShelfPlacement[] = []
  const layers: StyleLayer[] = []
  let right = 0
  let bayWidth = 0
  // Runs past `rows` levels go into more bays to the right. Each bay is a whole case of the style,
  // so the case keeps the stage's height and grows along the screen's x.
  for (let first = 0; first < layout.shelves.length; first += perBay) {
    const last = Math.min(layout.shelves.length, first + perBay)
    // Runs hold positions in layout.books; callers may renumber `index` (the browser repack does).
    const members = layout.books.flatMap((book, i) =>
      book.shelf >= first && book.shelf < last ? [i] : [],
    )
    const bay: BookcaseLayout<T> = {
      ...layout,
      shelves: layout.shelves.slice(first, last).map((shelf, i) => ({ ...shelf, index: i })),
      books: members.map(i => ({ ...layout.books[i], shelf: layout.books[i].shelf - first })),
    }
    const runs: number[][] = bay.shelves.map(() => [])
    bay.books.forEach((book, i) => runs[book.shelf].push(i))
    const built = STYLE_BUILDERS[style]({ layout: bay, pick, runs, showTags })
    const [minX, , maxX] = screenBounds(
      built.layers,
      bay.books.map(book => worldBook(book, built.placements[book.shelf])),
    )
    const dx = first === 0 ? 0 : right + BAY_GAP - minX
    if (first === 0) bayWidth = maxX - minX
    right = maxX + dx
    // World (d, d, 0) projects to (2d cos30, 0), a step along the screen's x only.
    const d = dx / (2 * COS30)
    for (const p of built.placements) {
      const [x, y, z] = p.origin
      placements.push({ ...p, shelf: p.shelf + first, origin: [x + d, y + d, z] })
    }
    for (const layer of built.layers) {
      const moved = dx === 0 ? layer : shiftLayer(layer, dx)
      layers.push(
        moved.kind === 'shelf'
          ? { ...moved, shelf: moved.shelf + first, books: moved.books.map(i => members[i]) }
          : moved,
      )
    }
  }
  const worldBooks = layout.books.map(book => worldBook(book, placements[book.shelf]))
  const bounds = screenBounds(layers, worldBooks)
  const caseWidth =
    (Math.max(layout.shelfWidth, BOOKCASE_DEFAULTS.shelfWidth) + 2 * WALL + 10 + layout.depth) *
    COS30
  return {
    ...layout,
    style,
    placements,
    worldBooks,
    layers,
    bounds,
    fit: Math.min(bayWidth, caseWidth),
  }
}
