// SVG strings for case layers, shelves and books. The strings hold geometry, class names and
// colour tokens only; note text is written afterwards with `fillText`, never through markup.
import {
  type DefNode,
  type Point,
  type SceneBook,
  type SceneHalo,
  type SceneLayer,
  type ScenePart,
  type ScenePlate,
  type Segment,
  type ShelfMaterial,
  faceFill,
  grainFill,
  materialCss,
  materialDefs,
  spineInk,
  toMatrix,
  toPath,
  toPoints,
} from './scene-model'

export interface Box {
  x: number
  y: number
  w: number
  h: number
}

export type ShelfLayer = Extract<SceneLayer, { kind: 'shelf' }>
export type SolidsLayer = Extract<SceneLayer, { kind: 'solids' }>

export interface BookPaint {
  /** Spine colour as drawn (CSS colour). */
  spine: string
  ink: string
  /** Draw the spine labels; false when the spine is too thin on screen. */
  text: boolean
}

const r1 = (n: number) => Math.round(n * 10) / 10

/** Spine labels are hidden below 9 px of spine on screen. */
export const spineLabels = <T>(book: SceneBook<T>, scale: number) => book.thickness * scale >= 9

export function paintBook<T>(book: SceneBook<T>, spine: string, scale: number): BookPaint {
  return { spine, ink: spineInk(spine), text: spineLabels(book, scale) }
}

export const boxOf = (points: readonly Point[], pad = 2): Box => boxOfLists([points], pad)

export const viewBox = (b: Box) => `${r1(b.x)} ${r1(b.y)} ${r1(b.w)} ${r1(b.h)}`

const polygon = (className: string, points: readonly Point[], attrs = '') =>
  `<polygon class="${className}" points="${toPoints(points)}"${attrs}/>`

const surfaces = (className: string, faces: readonly (readonly Point[])[]) => {
  if (faces.length === 0) return ''
  const d = faces
    .map(points => `M${points.map(([x, y]) => `${r1(x)} ${r1(y)}`).join('L')}Z`)
    .join('')
  return `<path class="${className}" d="${d}"/>`
}

const lines = (className: string, segments: readonly Segment[]) =>
  segments.length > 0 ? `<path class="${className}" d="${toPath(segments)}"/>` : ''

// Paper keeps the view's theme tokens; another material swaps them (`materialTokens`).
const material = (m: ShelfMaterial) => (m === 'paper' ? '' : ` data-material="${m}"`)

const halo = (h: SceneHalo) => polygon(`bc-halo bc-halo-${h.kind}`, h.points, material(h.material))

/** Faces with their finish and grain, then bevel, ply, highlight and form lines, creases, silhouette and screws. */
function part(p: ScenePart): string {
  let out = `<g class="bc-solid" data-part="${p.part}"${material(p.material)}>`
  // The tone class keeps the flat fill as the fallback; the finish gradient sits on the element.
  for (const face of p.faces)
    out += polygon(
      `bc-tone-${face.tone}`,
      face.points,
      ` style="fill:${faceFill(p.material, face)}"`,
    )
  for (const face of p.faces) {
    const grain = grainFill(p.material, face)
    if (grain) out += polygon('bc-grain', face.points, ` fill="${grain}"`)
  }
  for (const line of p.lines) out += lines(`bc-${line.kind}`, line.segments)
  out += lines('bc-crease', p.creases)
  if (p.outline) out += polygon('bc-edge', p.outline)
  out += lines('bc-edge', p.edges)
  if (p.screws.length > 0)
    out += `<g class="bc-screws">${p.screws
      .map(([x, y]) => `<ellipse cx="${r1(x)}" cy="${r1(y)}" rx="2.2" ry="1.27"/>`)
      .join('')}</g>`
  return `${out}</g>`
}

function plate(p: ScenePlate, shelf: number): string {
  return (
    polygon('bc-plate', p.face) +
    lines('bc-plate-ticks', p.ticks) +
    `<text class="bc-plate-label" transform="${toMatrix(p.matrix)}" dominant-baseline="central" data-text="p${shelf}"></text>`
  )
}

/** A case layer: halos under its parts. */
export function solidsMarkup(layer: SolidsLayer): string {
  return layer.halos.map(halo).join('') + layer.solids.map(part).join('')
}

/** Change key of halos and parts: their kinds, materials and the first point of each face. */
export function partsSignature(halos: readonly SceneHalo[], solids: readonly ScenePart[]): string {
  let out = ''
  for (const h of halos) out += `${h.kind}${h.material}${h.points[0]};`
  for (const p of solids) {
    out += `${p.part}${p.material}${p.faces.length}${p.lines.length}${p.screws.length}`
    for (const face of p.faces) out += `${face.tone}${face.points[0]}`
    out += ';'
  }
  return out
}

export function solidsBox(layer: SolidsLayer): Box {
  function* points(): Generator<readonly Point[]> {
    for (const h of layer.halos) yield h.points
    for (const s of layer.solids) for (const f of s.faces) yield f.points
  }
  return boxOfLists(points())
}

function bookBody<T>(book: SceneBook<T>, paint: BookPaint, cover: string): string {
  const { faces, detail, text, index } = book
  let labels = ''
  if (paint.text && text.title)
    labels += `<text class="bc-spine-title" font-size="${r1(text.title.size)}" data-text="t${index}"></text>`
  if (paint.text && text.author)
    labels += `<text class="bc-spine-author" x="${r1(text.author.x)}" font-size="${r1(text.author.size)}" text-anchor="end" data-text="a${index}"></text>`
  return (
    polygon('bc-face-spine', faces.spine) +
    polygon('bc-spine-sheen', faces.spine, ` fill="${book.sheen}"`) +
    `<g class="bc-jacket">${polygon('bc-face-cover', faces.cover)}${cover}</g>` +
    polygon('bc-face-end', faces.end) +
    polygon('bc-pages', detail.pages) +
    lines('bc-page-crease', detail.pageCreases) +
    surfaces('bc-board-rim', detail.boardRims) +
    surfaces('bc-jacket-rim', detail.jacketRims) +
    lines('bc-jacket-fold', detail.jacketCreases) +
    lines('bc-round-lit', [detail.round[0]]) +
    lines('bc-round-shade', [detail.round[1]]) +
    lines('bc-crease', faces.creases) +
    polygon('bc-edge', faces.edge) +
    `<g class="bc-spine-text" transform="${toMatrix(text.matrix)}" dominant-baseline="central">${labels}</g>`
  )
}

export const bookStyle = (paint: BookPaint) => `--spine:${paint.spine};--spine-ink:${paint.ink}`

export function bookMarkup<T>(book: SceneBook<T>, paint: BookPaint): string {
  return `<g class="bc-book" data-book="${book.index}" style="${bookStyle(paint)}">${bookBody(book, paint, '')}</g>`
}

/**
 * The lifted book at full colour, with its cover under the edge: on the front, an `<image>` (its
 * href is set by the caller) or the title; on a visible back cover, the title only.
 */
export function liftMarkup<T>(
  book: SceneBook<T>,
  paint: BookPaint,
  cover: { image: boolean; ink: string; lines: number },
): string {
  let face = ''
  const frame = book.cover ?? book.back
  if (frame) {
    const w = r1(book.depth)
    const h = r1(book.height)
    const y = r1(book.height * 0.3 - (cover.lines - 1) * 8)
    const content =
      cover.image && book.cover
        ? `<image class="bc-cover-image" width="${w}" height="${h}" preserveAspectRatio="xMidYMid slice"/>`
        : `<text class="bc-cover-title" x="${r1(book.depth / 2)}" y="${y}" font-size="12" text-anchor="middle" style="fill:${cover.ink}" data-text="c${book.index}"></text>`
    face = `<g class="bc-cover" transform="${toMatrix(frame)}">${content}</g>`
  }
  return `<g class="bc-book is-lit" data-book="${book.index}" style="${bookStyle(paint)}">${bookBody(book, paint, face)}</g>`
}

export function bookBox<T>(book: SceneBook<T>): Box {
  return boxOf(book.faces.edge)
}

/** Board, plate, contact halo, then the books in the order given (SPEC 5.2 step 2). */
export function shelfMarkup<T>(
  layer: ShelfLayer,
  books: readonly SceneBook<T>[],
  paint: (book: SceneBook<T>) => BookPaint,
): string {
  let out = layer.halos.map(halo).join('') + layer.solids.map(part).join('')
  if (layer.plate) out += plate(layer.plate, layer.shelf)
  out += halo(layer.contact)
  for (const index of layer.books) out += bookMarkup(books[index], paint(books[index]))
  return out
}

export function shelfBox<T>(layer: ShelfLayer, books: readonly SceneBook<T>[]): Box {
  function* points(): Generator<readonly Point[]> {
    yield layer.contact.points
    if (layer.plate) yield layer.plate.face
    for (const h of layer.halos) yield h.points
    for (const s of layer.solids) for (const f of s.faces) yield f.points
    for (const index of layer.books) yield books[index].faces.edge
  }
  return boxOfLists(points())
}

function boxOfLists(lists: Iterable<readonly Point[]>, pad = 2): Box {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const list of lists)
    for (const [x, y] of list) {
      if (x < x0) x0 = x
      if (y < y0) y0 = y
      if (x > x1) x1 = x
      if (y > y1) y1 = y
    }
  if (!Number.isFinite(x0)) return { x: 0, y: 0, w: 0, h: 0 }
  return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad }
}

/** Writes note text into the `data-text` slots of freshly set markup. */
export function fillText(root: Element, text: (key: string) => string | string[]): void {
  for (const el of Array.from(root.querySelectorAll('text[data-text]'))) {
    const value = text(el.getAttribute('data-text') ?? '')
    if (typeof value === 'string') {
      el.textContent = value
      continue
    }
    el.empty()
    const x = el.getAttribute('x') ?? '0'
    value.forEach((line, i) => {
      el.createSvg('tspan', { attr: { x, dy: i === 0 ? '0' : '1.3em' } }).textContent = line
    })
  }
}

const defMarkup = (node: DefNode): string => {
  const attrs = Object.entries(node.attrs)
    .map(([key, value]) => ` ${key}="${value}"`)
    .join('')
  return `<${node.tag}${attrs}>${(node.children ?? []).map(defMarkup).join('')}</${node.tag}>`
}

/**
 * Ladder tokens, gradient stops and the finish, grain and sheen defs for `materials`; veneer
 * tiles load from `assets`. The stops
 * resolve where they are defined, so this goes inside `.bookcase-world`; other layers reference
 * the defs by id.
 */
export function materialMarkup(materials: Iterable<ShelfMaterial>, assets: string): string {
  const set = new Set<ShelfMaterial>(['paper', ...materials])
  const css = materialCss(set, '.bookcase-world', '.theme-dark ')
  return `<style>${css}</style><defs>${materialDefs(set, assets).map(defMarkup).join('')}</defs>`
}
