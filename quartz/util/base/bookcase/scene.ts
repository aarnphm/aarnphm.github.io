import type { Element } from 'hast'
import { s } from 'hastscript'
import {
  type BookcaseLayout,
  type PlacedBook,
  type Point,
  type Segment,
  type Solid,
  COS30,
  bookDetail,
  bookSolid,
  caseFaces,
  coverMatrix,
  pullOffset,
  shelfFaces,
  spineCentre,
  spineText,
  toMatrix,
  toPath,
  toPoints,
} from './geometry'

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
}

const r1 = (n: number) => Math.round(n * 10) / 10

const polygon = (className: string, points: readonly Point[]): Element =>
  s(`polygon.${className}`, { points: toPoints(points) })

const lines = (className: string, segments: readonly Segment[]): Element =>
  s(`path.${className}`, { d: toPath(segments) })

/** Front, side and top fills, then creases, then the silhouette edge. */
function solid(className: string, parts: Solid, extra: Element[] = []): Element {
  return s(`g.bc-solid.${className}`, [
    polygon('bc-front', parts.front),
    polygon('bc-side', parts.side),
    polygon('bc-top', parts.top),
    ...extra,
    lines('bc-crease', parts.creases),
    polygon('bc-edge', parts.edge),
  ])
}

function bookNode(book: PlacedBook<SceneBook>): Element {
  const { item } = book
  const parts = bookSolid(book)
  const detail = bookDetail(book)
  const label = spineText(book, item.title, item.author)
  const [cx, cy] = spineCentre(book)
  const [px, py] = pullOffset(book)
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
      'data-book-index': book.index,
      'data-base-slug': item.slug,
      'data-cx': r1(cx),
      'data-cy': r1(cy),
      'data-cover': item.cover,
      'data-bookcase-cover': item.coverSlug,
      'data-spine-sampled': item.sampleSpine && item.coverSlug ? '' : undefined,
      'data-cover-transform': toMatrix(coverMatrix(book)),
      'data-cover-size': `${book.depth} ${book.height}`,
      style: `--spine:${item.spine};--spine-ink:${item.ink};--pull-x:${r1(px)}px;--pull-y:${r1(py)}px`,
    },
    [
      polygon('bc-front', parts.front),
      polygon('bc-side', parts.side),
      polygon('bc-top', parts.top),
      polygon('bc-pages', detail.pages),
      lines('bc-page-crease', detail.pageCreases),
      lines('bc-round-lit', [detail.round[0]]),
      lines('bc-round-shade', [detail.round[1]]),
      ...(detail.bands.length > 0 ? [lines('bc-band', detail.bands)] : []),
      lines('bc-crease', parts.creases),
      polygon('bc-edge', parts.edge),
      s(
        'g.bc-spine-text',
        { transform: toMatrix(label.matrix), 'dominant-baseline': 'central' },
        text,
      ),
    ],
  )
}

/**
 * Static isometric case. Painter's order: halo, plinth, back panel and left wall inner face; then
 * each shelf from the bottom up (board, plate, contact halo, books by ascending x); then the right
 * wall, the left wall's front edge and the crown. A pulled book is drawn later, in its own layer.
 */
export function bookcaseScene(layout: BookcaseLayout<SceneBook>, label: string): Element {
  const [minX, minY, maxX, maxY] = layout.bounds
  const pad = 16
  const width = r1(maxX - minX + pad * 2)
  const height = r1(maxY - minY + pad * 2)
  const byShelf = new Map<number, PlacedBook<SceneBook>[]>()
  for (const book of layout.books) {
    const list = byShelf.get(book.shelf) ?? []
    list.push(book)
    byShelf.set(book.shelf, list)
  }

  const parts = caseFaces(layout)
  const children: Element[] = [
    polygon('bc-halo.bc-halo-case', parts.halo),
    solid('bc-plinth', parts.plinth),
    polygon('bc-back', parts.back),
    polygon('bc-wall-inner', parts.leftInner),
  ]

  for (const shelf of [...layout.shelves].reverse()) {
    const faces = shelfFaces(shelf, layout)
    const plate = faces.plate
      ? [
          polygon('bc-plate', faces.plate.face),
          lines('bc-plate-ticks', faces.plate.ticks),
          s(
            'text.bc-plate-label',
            { transform: toMatrix(faces.plate.matrix), 'dominant-baseline': 'central' },
            faces.plate.label,
          ),
        ]
      : []
    children.push(
      s('g.bc-shelf', { 'data-shelf': shelf.index }, [
        solid('bc-board', faces.board, [lines('bc-bevel', [faces.bevel])]),
        ...plate,
        polygon('bc-halo.bc-halo-contact', faces.contact),
        ...(byShelf.get(shelf.index) ?? []).map(bookNode),
      ]),
    )
  }

  children.push(
    solid('bc-wall', parts.rightWall, [lines('bc-bevel', [parts.rightBevel])]),
    polygon('bc-wall-front', parts.leftFront),
    lines('bc-bevel', [parts.leftBevel]),
    polygon('bc-edge', parts.leftFront),
    solid('bc-crown', parts.crown, [
      lines('bc-bevel', [parts.crownBevel]),
      polygon('bc-inset', parts.crownInset),
      s(
        'g.bc-screws',
        parts.screws.map(([x, y]) => s('ellipse', { cx: r1(x), cy: r1(y), rx: 2.2, ry: 1.27 })),
      ),
    ]),
  )

  return s(
    'svg.bookcase-scene',
    {
      viewBox: `${r1(minX - pad)} ${r1(minY - pad)} ${width} ${height}`,
      width,
      height,
      role: 'img',
      'aria-label': label,
      'data-origin': `${r1(minX - pad)} ${r1(minY - pad)}`,
      // Screen width of the case with its walls, which the camera fits to the stage.
      'data-fit': r1((layout.shelfWidth + 34 + layout.depth) * COS30),
    },
    children,
  )
}
