import type { Element } from 'hast'
import { s } from 'hastscript'
import { type PlacedBook, type Point, type Segment, toMatrix, toPath, toPoints } from './geometry'
import {
  type ShelfMaterial,
  type StyleHalo,
  type StyleSolid,
  type StyledLayout,
  type WorldBook,
  bookDetailW,
  bookFacesW,
  coverMatrixW,
  materialLadder,
  pullOffsetW,
  spineCentreW,
  spineMatrixW,
  spineTextW,
} from './styles'

export interface SceneBook {
  slug: string
  title: string
  author?: string
  spine: string
  ink: string
  /** Page-relative cover URL; the emitter may replace it with a thumbnail. */
  cover?: string
  /** Content slug of the cover image, for the emitter. */
  coverSlug?: string
  /** True when the spine colour should come from the cover. */
  sampleSpine: boolean
  /** Verified local edition spine derivative. */
  spineArt?: string
  /** Larger verified derivative, loaded only for the selected book. */
  spineArtDetail?: string
  spineArtDetailWidth?: number
  spineArtDetailHeight?: number
}

const r1 = (n: number) => Math.round(n * 10) / 10

const polygon = (className: string, points: readonly Point[]): Element =>
  s(`polygon.${className}`, { points: toPoints(points) })

const strokes = (className: string, segments: readonly Segment[]): Element =>
  s(`path.${className}`, { d: toPath(segments) })

const materialAttr = (material: ShelfMaterial) => (material === 'paper' ? undefined : material)

function solidNode(solid: StyleSolid): Element {
  const children: Element[] = solid.faces.map(face => polygon(`bc-tone-${face.tone}`, face.points))
  for (const line of solid.lines) children.push(strokes(`bc-${line.kind}`, line.segments))
  if (solid.creases.length > 0) children.push(strokes('bc-crease', solid.creases))
  if (solid.outline) children.push(polygon('bc-edge', solid.outline))
  if (solid.edges.length > 0) children.push(strokes('bc-edge', solid.edges))
  if (solid.screws.length > 0) {
    children.push(
      s(
        'g.bc-screws',
        solid.screws.map(([x, y]) => s('ellipse', { cx: r1(x), cy: r1(y), rx: 2.2, ry: 1.27 })),
      ),
    )
  }
  return s(
    'g.bc-solid',
    { 'data-part': solid.part, 'data-material': materialAttr(solid.material) },
    children,
  )
}

const haloNode = (halo: StyleHalo): Element =>
  s(`polygon.bc-halo.bc-halo-${halo.kind}`, {
    points: toPoints(halo.points),
    'data-material': materialAttr(halo.material),
  })

/** Light and dark ladders of the materials in use; paper keeps the theme tokens. */
function materialStyle(materials: Set<ShelfMaterial>): Element | null {
  const rules: string[] = []
  for (const material of materials) {
    if (material === 'paper') continue
    for (const theme of ['light', 'dark'] as const) {
      const ladder = materialLadder(material, theme)
      const tokens = Object.entries(ladder)
        .filter(([, value]) => value)
        .map(([key, value]) => `--bc-${key}:${value}`)
        .join(';')
      const root = theme === 'dark' ? ":root[saved-theme='dark'] " : ''
      rules.push(`${root}.bookcase-scene [data-material='${material}']{${tokens}}`)
    }
  }
  return rules.length > 0 ? s('style', rules.join('')) : null
}

function bookNode(placed: PlacedBook<SceneBook>, world: WorldBook): Element {
  const book = placed.item
  const faces = bookFacesW(world)
  const detail = bookDetailW(world)
  const label = spineTextW(world, book.title, book.author)
  const [cx, cy] = spineCentreW(world)
  const [px, py] = pullOffsetW(world)
  const cover = coverMatrixW(world)
  const text: Element[] = []
  if (label.title)
    text.push(s('text.bc-spine-title', { 'font-size': r1(label.title.size) }, label.title.text))
  if (label.author)
    text.push(
      s(
        'text.bc-spine-author',
        { x: r1(label.author.x), 'font-size': r1(label.author.size), 'text-anchor': 'end' },
        label.author.text,
      ),
    )

  return s(
    'g.bc-book',
    {
      'data-book-index': placed.index,
      'data-book-path': placed.path,
      'data-title': book.title,
      'data-author': book.author,
      'data-base-slug': book.slug,
      'data-cx': r1(cx),
      'data-cy': r1(cy),
      'data-cover': book.cover,
      'data-full-cover': book.cover,
      'data-bookcase-cover': book.coverSlug,
      'data-spine-sampled': book.sampleSpine && book.coverSlug ? '' : undefined,
      // A rot-90 book shows its back cover, so it gets no cover image.
      'data-cover-transform': cover ? toMatrix(cover) : undefined,
      'data-cover-size': `${world.depth} ${world.height}`,
      'data-spine-art': book.spineArt,
      'data-spine-art-detail': book.spineArtDetail,
      'data-spine-art-detail-width': book.spineArtDetailWidth,
      'data-spine-art-detail-height': book.spineArtDetailHeight,
      'data-spine-transform': toMatrix(spineMatrixW(world)),
      'data-spine-size': `${world.thickness} ${world.height}`,
      style: `--spine:${book.spine};--spine-ink:${book.ink};--pull-x:${r1(px)}px;--pull-y:${r1(py)}px`,
    },
    [
      polygon('bc-face-spine', faces.spine),
      s('g.bc-jacket', [polygon('bc-face-cover', faces.cover)]),
      polygon('bc-face-end', faces.end),
      polygon('bc-pages', detail.pages),
      strokes('bc-page-crease', detail.pageCreases),
      ...detail.boardRims.map(points => polygon('bc-board-rim', points)),
      ...detail.jacketRims.map(points => polygon('bc-jacket-rim', points)),
      strokes('bc-jacket-fold', detail.jacketCreases),
      strokes('bc-round-lit', [detail.round[0]]),
      strokes('bc-round-shade', [detail.round[1]]),
      strokes('bc-crease', faces.creases),
      polygon('bc-edge', faces.edge),
      s(
        'g.bc-spine-text',
        { transform: toMatrix(label.matrix), 'dominant-baseline': 'central' },
        text,
      ),
    ],
  )
}

/** Static scene in the layout's painter's order. A pulled book is drawn later, in its own layer. */
export function bookcaseScene(layout: StyledLayout<SceneBook>, label: string): Element {
  const [minX, minY, maxX, maxY] = layout.bounds
  const pad = 16
  const width = r1(maxX - minX + pad * 2)
  const height = r1(maxY - minY + pad * 2)
  const materials = new Set<ShelfMaterial>()
  const children: Element[] = []
  for (const layer of layout.layers) {
    if (layer.kind === 'solids') {
      for (const halo of layer.halos) children.push(haloNode(halo))
      for (const solid of layer.solids) {
        materials.add(solid.material)
        children.push(solidNode(solid))
      }
      continue
    }
    const plate = layer.plate
      ? [
          polygon('bc-plate', layer.plate.face),
          strokes('bc-plate-ticks', layer.plate.ticks),
          s(
            'text.bc-plate-label',
            { transform: toMatrix(layer.plate.matrix), 'dominant-baseline': 'central' },
            layer.plate.label,
          ),
        ]
      : []
    children.push(
      s('g.bc-shelf', { 'data-shelf': layer.shelf }, [
        ...plate,
        haloNode(layer.contact),
        ...layer.books.map(index => bookNode(layout.books[index], layout.worldBooks[index])),
      ]),
    )
  }
  const style = materialStyle(materials)

  return s(
    'svg.bookcase-scene',
    {
      viewBox: `${r1(minX - pad)} ${r1(minY - pad)} ${width} ${height}`,
      width,
      height,
      role: 'img',
      'aria-label': label,
      'data-origin': `${r1(minX - pad)} ${r1(minY - pad)}`,
      'data-style': layout.style,
    },
    style ? [style, ...children] : children,
  )
}
