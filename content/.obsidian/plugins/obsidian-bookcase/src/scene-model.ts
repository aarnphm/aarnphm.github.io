// The only module that calls the shared geometry and styles. It turns a packed and placed layout
// into painter's-order layers and per-book records; the painters (svg, stage, lift) read only
// these. Parts, halos and plates keep the garden's StyleSolid, StyleHalo and Plate shapes.
import {
  type Matrix,
  type PackOptions,
  type Point,
  type Segment,
  type ShelfGroup,
  COS30,
  packShelves,
  project,
} from '../../../../../quartz/util/base/bookcase/geometry'
import {
  type BookDetail,
  type BookcaseStyle,
  type Plate,
  type ShelfMaterial,
  type SpineText,
  type StyleHalo,
  type StyleSolid,
  type WorldBook,
  bookDetailW,
  bookFacesW,
  coverMatrixW,
  placeShelves,
  pullOffsetW,
  sheenFill,
  spineCentreW,
  spineTextW,
  stylePackOptions,
} from '../../../../../quartz/util/base/bookcase/styles'

export type { Matrix, Point, Segment, ShelfGroup, SpineText }
export type { BookcaseStyle, ShelfMaterial }
export type ScenePart = StyleSolid
export type SceneHalo = StyleHalo
export type ScenePlate = Plate
// Constants, colour and formatting helpers the painters and options share with the garden.
export {
  BOOKCASE_DEFAULTS,
  BOOKCASE_LIMITS,
  INK_DARK,
  INK_LIGHT,
  clampOption,
  fallbackSpine,
  hashPath,
  rgbToHex,
  spineInk,
  toMatrix,
  toPath,
  toPoints,
} from '../../../../../quartz/util/base/bookcase/geometry'
export {
  BOOKCASE_STYLES,
  SHELF_MATERIALS,
  faceFill,
  grainFill,
  hasTopicTag,
  materialCss,
  materialDefs,
  topicKeys,
  topicLabel,
  topicShelf,
  topicShortcuts,
} from '../../../../../quartz/util/base/bookcase/styles'
export type { DefNode, TopicShelf } from '../../../../../quartz/util/base/bookcase/styles'

/**
 * Painter's order. A shelf layer is one windowed `<svg>`: the solids layer that comes right
 * before it in the garden's order (its board, blade or leaf), then its label plate, contact halo
 * and books in the order given. Any other solids layer is a static case `<svg>`. The lifted book
 * is painted after every layer.
 */
export type SceneLayer =
  | { kind: 'solids'; halos: SceneHalo[]; solids: ScenePart[] }
  | {
      kind: 'shelf'
      shelf: number
      halos: SceneHalo[]
      solids: ScenePart[]
      plate: ScenePlate | null
      contact: SceneHalo
      books: number[]
      /** Run origin, for change detection. */
      origin: readonly [number, number, number]
    }

export interface SceneBook<T> {
  index: number
  path: string
  shelf: number
  item: T
  thickness: number
  height: number
  depth: number
  /** Spine labels as given, before the geometry fits them. */
  label: SpineLabel
  faces: { spine: Point[]; cover: Point[]; end: Point[]; creases: Segment[]; edge: Point[] }
  /** Camera target. */
  centre: Point
  // Computed on first read: only mounted shelves and the lift need them.
  readonly detail: BookDetail
  /** Screen offset of the pulled book. */
  readonly pull: Point
  readonly text: SpineText
  /** Gradient reference that rounds the spine, lit along its lit edge. */
  readonly sheen: string
  /** Frame of a `depth × height` front-cover image; null when the visible side is the back. */
  readonly cover: Matrix | null
  /** Frame of the visible back cover (rot-90 wing), for its title; null otherwise. */
  readonly back: Matrix | null
}

class PlacedBook<T> implements SceneBook<T> {
  readonly faces: SceneBook<T>['faces']
  readonly centre: Point
  private cache: {
    detail?: BookDetail
    pull?: Point
    text?: SpineText
    cover?: Matrix | null
    back?: Matrix | null
  } = {}

  constructor(
    readonly index: number,
    readonly path: string,
    readonly shelf: number,
    readonly item: T,
    readonly label: SpineLabel,
    private world: WorldBook,
  ) {
    this.faces = bookFacesW(world)
    this.centre = spineCentreW(world)
  }

  get thickness(): number {
    return this.world.thickness
  }

  get height(): number {
    return this.world.height
  }

  get depth(): number {
    return this.world.depth
  }

  get detail(): BookDetail {
    return (this.cache.detail ??= bookDetailW(this.world))
  }

  get pull(): Point {
    return (this.cache.pull ??= pullOffsetW(this.world))
  }

  get sheen(): string {
    return sheenFill(this.world)
  }

  get text(): SpineText {
    return (this.cache.text ??= spineTextW(this.world, this.label.title, this.label.author))
  }

  get cover(): Matrix | null {
    if (this.cache.cover === undefined) this.cache.cover = coverMatrixW(this.world)
    return this.cache.cover
  }

  get back(): Matrix | null {
    if (this.cache.back === undefined) this.cache.back = backFrame(this.world)
    return this.cache.back
  }
}

export interface Scene<T> {
  style: BookcaseStyle
  books: SceneBook<T>[]
  layers: SceneLayer[]
  /** Group label of each run. */
  labels: (string | null)[]
  /** Screen bounds of every part and every pulled book: [minX, minY, maxX, maxY]. */
  bounds: readonly [number, number, number, number]
  /** Screen width the camera fits to the stage (the garden's rule). */
  fit: number
  /** Screen x range of the case and its books at rest, without the pulled positions. */
  span: readonly [number, number]
}

export interface SpineLabel {
  title: string
  author?: string
}

export interface SceneOptions extends PackOptions {
  style: BookcaseStyle
  /** One material for every part; null keeps the style's own materials. */
  material: ShelfMaterial | null
  /** False drops every label plate. */
  showTags: boolean
}

/** Back cover of a rot-90 book: the -y face, read with the spine on the right. */
function backFrame(b: WorldBook): Matrix | null {
  if (b.rot !== 90) return null
  const [e, f] = project(b.x, b.y, b.z + b.h)
  return [COS30, 0.5, 0, 1, e, f]
}

export function buildScene<T>(
  groups: ShelfGroup<T>[],
  options: SceneOptions,
  label: (item: T) => SpineLabel,
): Scene<T> {
  const count = groups.reduce((n, g) => n + g.books.length, 0)
  const pack = stylePackOptions(
    options.style,
    {
      bookHeight: options.bookHeight,
      shelfWidth: options.shelfWidth,
      continuous: options.continuous,
    },
    count,
  )
  const layout = placeShelves(
    packShelves(groups, pack),
    options.style,
    options.material ?? undefined,
    { showTags: options.showTags },
  )

  const layers: SceneLayer[] = []
  for (const layer of layout.layers) {
    if (layer.kind === 'shelf') {
      const own = layers.at(-1)
      const merged = own?.kind === 'solids'
      if (merged) layers.pop()
      layers.push({
        ...layer,
        halos: merged ? own.halos : [],
        solids: merged ? own.solids : [],
        origin: layout.placements[layer.shelf].origin,
      })
    } else layers.push(layer)
  }

  const books = layout.books.map(
    (placed, i): SceneBook<T> =>
      new PlacedBook(
        i,
        placed.path,
        placed.shelf,
        placed.item,
        label(placed.item),
        layout.worldBooks[i],
      ),
  )

  let left = Infinity
  let right = -Infinity
  const reach = (points: readonly Point[]) => {
    for (const [x] of points) {
      if (x < left) left = x
      if (x > right) right = x
    }
  }
  for (const layer of layout.layers)
    if (layer.kind === 'solids') {
      for (const solid of layer.solids) for (const face of solid.faces) reach(face.points)
      for (const halo of layer.halos) reach(halo.points)
    } else {
      reach(layer.contact.points)
      if (layer.plate) reach(layer.plate.face)
    }
  for (const book of books) reach(book.faces.edge)

  return {
    style: layout.style,
    books,
    layers,
    labels: layout.shelves.map(shelf => shelf.label),
    bounds: layout.bounds,
    fit: layout.fit,
    span: left <= right ? [left, right] : [layout.bounds[0], layout.bounds[2]],
  }
}
