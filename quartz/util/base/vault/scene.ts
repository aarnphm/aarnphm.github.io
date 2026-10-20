import type { Element } from 'hast'
import { s } from 'hastscript'
import {
  COS30,
  type Point,
  box,
  fitText,
  hashPath,
  project,
  toPath,
  toPoints,
} from '../bookcase/geometry'
import {
  type PlacedCan,
  type VaultLayout,
  HOLDER_CHAR,
  HOLDER_PAD,
  RACK,
  bandPath,
  bayStart,
  discMatrix,
  pullOffset,
  rackHeight,
  rimCentre,
  rimPoint,
  shelfTop,
} from './layout'
import {
  type VaultMaterial,
  type VaultTone,
  hasGrain,
  toneFill,
  vaultCss,
  vaultDefs,
} from './styles'

export type LabelKind = 'vinyl' | 'tape' | 'viewed'

export interface SceneCan {
  title: string
  /** Director surnames, joined. */
  surname: string
  year?: number
  rating?: number
  kind: LabelKind
  watched?: string
  studio?: string
}

const r1 = (n: number) => Math.round(n * 10) / 10

/** Screen length of the 150° to 210° label arc of radius r. */
function arcLength(r: number): number {
  let length = 0
  let last: Point | undefined
  for (let deg = 150; deg <= 210; deg += 5) {
    const t = (deg * Math.PI) / 180
    const p = project(0, r * Math.cos(t), r * Math.sin(t))
    if (last) length += Math.hypot(p[0] - last[0], p[1] - last[1])
    last = p
  }
  return length
}

const ARC = { 1000: { r: 135, length: arcLength(135) }, 2000: { r: 190, length: arcLength(190) } }
/** Title and second-line sizes, and each line's offset from the strip edge along +x. */
const TYPE = {
  1000: { title: 9.5, line: 7, titleAt: 8, lineAt: 19 },
  2000: { title: 11.5, line: 8, titleAt: 8, lineAt: 22 },
}
/** Berkeley Mono advance. */
const EM = 0.6
const START = 4

/** Required parts, then optional parts while they fit, then the tail (the rating) always. */
function composeLine(required: string[], optional: string[], tail: string, max: number): string {
  const parts = [...required]
  const fits = (list: string[]) => [...list, tail].filter(Boolean).join(' · ').length <= max
  for (const part of optional) if (fits([...parts, part])) parts.push(part)
  const line = [...parts, tail].filter(Boolean).join(' · ')
  return line.length <= max ? line : fitText(line, max, 1, 1)
}

function labelLines(can: SceneCan, footage: 1000 | 2000): { title: string; line: string } {
  const type = TYPE[footage]
  const room = ARC[footage].length - START * 2
  const rating = can.rating === undefined ? '' : can.rating.toFixed(1)
  const year = can.year === undefined ? '' : String(can.year)
  const surname = can.surname.toUpperCase()
  const max = Math.floor(room / (type.line * EM))
  let line: string
  if (can.kind === 'tape') {
    const short = year ? `'${year.slice(-2)}` : ''
    line = composeLine([[surname, short].filter(Boolean).join(' ')], [], rating, max)
  } else if (can.kind === 'viewed' && can.watched) {
    line = composeLine([`VIEWED ${can.watched}`], [], rating, max)
  } else {
    const optional = [
      ...(footage === 2000 && can.studio ? [can.studio.toUpperCase()] : []),
      '35 MM',
      `${footage} FT`,
    ]
    line = composeLine(
      [surname ? `DIR. ${surname}` : '', year].filter(Boolean),
      optional,
      rating,
      max,
    )
  }
  return { title: fitText(can.title.toUpperCase(), room, type.title, EM), line }
}

const polygon = (cls: string, points: readonly Point[], fill?: string): Element =>
  s('polygon', { class: cls, points: toPoints(points), fill })

/** A box solid in the anatomy idiom: three gradient faces and one path of edges and creases. */
function boxNodes(
  material: VaultMaterial,
  [x, y, z]: readonly [number, number, number],
  [w, d, h]: readonly [number, number, number],
  tones: readonly [VaultTone, VaultTone, VaultTone] = ['face', 's2', 'top'],
): Element[] {
  const solid = box(x, y, z, w, d, h)
  return [
    polygon('cn-face', solid.front, toneFill(material, tones[0])),
    polygon('cn-face', solid.side, toneFill(material, tones[1])),
    polygon('cn-face', solid.top, toneFill(material, tones[2])),
    s('path', {
      class: `cn-edge cn-m-${material}`,
      d: `M${toPoints(solid.edge).replaceAll(' ', 'L')}Z${toPath(solid.creases)}`,
    }),
  ]
}

function canNode(can: PlacedCan<SceneCan>, rowEnd: boolean): Element {
  const { item, footage, finish, thickness: w } = can
  const r = can.diameter / 2
  const x0 = can.x
  const type = TYPE[footage]
  const { title, line } = labelLines(item, footage)
  const band = bandPath(can, x0, x0 + w, 45, 225)
  const [cx, cy] = rimCentre(can)
  const [px, py] = pullOffset(can)
  const tape = item.kind === 'tape'
  const tilt = tape ? r1(((hashPath(`${can.path}#s`) % 1000) / 1000) * 4 - 2) : 0
  const rotate = tilt ? `rotate(${tilt} ${r1(cx)} ${r1(cy)}) ` : ''
  const k = Math.round((r / ARC[footage].r) * 1000) / 1000
  const text = (cls: string, at: number, value: string, size: number) => {
    const [e, f] = project(x0 + 6 + at, can.y + r, can.z + r)
    return s(
      'text',
      {
        class: cls,
        'font-size': size,
        transform: `${rotate}translate(${r1(e)} ${r1(f)}) scale(${k})`,
      },
      [s('textPath', { href: `#cn-arc-${ARC[footage].r}`, startOffset: START }, value)],
    )
  }
  const children: Element[] = [
    s('path', { class: 'cn-band', d: band, fill: toneFill(finish, 'face') }),
  ]
  if (hasGrain(finish))
    children.push(s('path', { class: 'cn-grain', d: band, fill: `url(#cn-grain-${finish})` }))
  children.push(
    s('path', {
      class: tape ? 'cn-strip cn-tape' : 'cn-strip',
      d: bandPath(can, x0 + 6, x0 + w - 6, 150, 210),
      transform: tilt ? `rotate(${tilt} ${r1(cx)} ${r1(cy)})` : undefined,
    }),
  )
  if (title) children.push(text('cn-title', type.titleAt, title, type.title))
  if (line)
    children.push(
      text(
        item.kind === 'viewed' && item.watched ? 'cn-line cn-stamp' : 'cn-line',
        type.lineAt,
        line,
        type.line,
      ),
    )
  const rating = item.rating ?? 0
  if (rating >= 7.5) {
    const [dx, dy] = rimPoint(can, x0 + w / 2, 120)
    children.push(
      s('ellipse', {
        class: rating >= 8.5 ? 'cn-dot cn-dot-high' : 'cn-dot cn-dot-good',
        cx: r1(dx),
        cy: r1(dy),
        rx: footage === 2000 ? 7 : 6,
        ry: footage === 2000 ? 5 : 4,
      }),
    )
  }
  const lid = discMatrix(can, x0 + w)
  children.push(s('circle', { class: 'cn-lid', r, transform: lid, fill: toneFill(finish, 's2') }))
  if (rowEnd) children.push(s('circle', { class: 'cn-rings', r, transform: lid }))
  // Only the front of the roll, the face that carries the label, takes the pointer.
  children.push(s('path', { class: 'cn-hit', d: bandPath(can, x0, x0 + w, 135, 225) }))
  return s(
    'g',
    {
      class: `cn-can cn-m-${finish}`,
      'data-index': can.index,
      'data-cx': r1(cx),
      'data-cy': r1(cy),
      'data-r': r,
      style: `--px:${r1(px)}px;--py:${r1(py)}px`,
    },
    children,
  )
}

function dividerNodes(material: VaultMaterial, x: number, z: number, height: number): Element[] {
  const solid = box(x, 10, z, RACK.divider, RACK.depth - 20, height)
  return [
    polygon('cn-face', solid.edge, toneFill(material, 's2')),
    s('path', {
      class: `cn-edge cn-m-${material}`,
      d: `M${toPoints(solid.edge).replaceAll(' ', 'L')}Z${toPath(solid.creases)}`,
    }),
  ]
}

function holderNodes(
  material: VaultMaterial,
  x: number,
  end: number,
  z: number,
  label: string,
): Element[] {
  const x1 = Math.min(end, x + label.length * HOLDER_CHAR + HOLDER_PAD)
  const text = fitText(label, x1 - x - HOLDER_PAD + 1, HOLDER_CHAR, 1)
  if (!text) return []
  const q = (lx: number, lz: number) => project(lx, -1.2, lz)
  const [z0, z1] = [z - RACK.board + 3, z - 3]
  const [e, f] = q(x + 8, (z0 + z1) / 2)
  return [
    polygon(`cn-holder cn-m-${material}`, [q(x, z0), q(x1, z0), q(x1, z1), q(x, z1)]),
    s(
      'text',
      {
        class: 'cn-holder-label',
        transform: `matrix(${COS30.toFixed(4)} 0.5 0 1 ${r1(e)} ${r1(f)})`,
        'dominant-baseline': 'central',
      },
      text,
    ),
  ]
}

function crankNode(x: number, z: number): Element {
  const [e, f] = project(x, RACK.depth / 2 + 10, z)
  const spokes = [90, 210, 330]
    .map(deg => {
      const t = (deg * Math.PI) / 180
      return `M${r1(22 * Math.cos(t))} ${r1(22 * Math.sin(t))}L${r1(104 * Math.cos(t))} ${r1(104 * Math.sin(t))}`
    })
    .join('')
  return s(
    'g',
    {
      class: 'cn-crank cn-m-black-lacquer',
      transform: `matrix(${COS30.toFixed(4)} -0.5 0 1 ${r1(e)} ${r1(f)})`,
    },
    [
      s('circle', { class: 'cn-crank-rim', r: 110 }),
      s('path', { class: 'cn-crank-spokes', d: spokes }),
      s('circle', { class: 'cn-crank-hub', r: 22 }),
      s('circle', { class: 'cn-crank-knob', r: 12, cx: 78, cy: -78 }),
    ],
  )
}

export interface SceneOptions {
  material: VaultMaterial
  showTags: boolean
  crank: boolean
  label: string
}

/** The rack in painter's order; the script paints a pulled can last, in `.vault-lift`. */
export function vaultScene(layout: VaultLayout<SceneCan>, options: SceneOptions): Element {
  const { bayWidth: W, bayShelves: S, bays } = layout
  const steel = options.material
  const [minX, minY, maxX, maxY] = layout.bounds
  const pad = 24
  const width = r1(maxX - minX + pad * 2)
  const height = r1(maxY - minY + pad * 2)
  const carriage = bayStart(bays, W)
  const tall = rackHeight(S)
  const rowEnds = new Set(layout.rowEnds)
  const byShelf = new Map<number, { x: number; node: Element[] }[]>()
  const place = (shelf: number, x: number, node: Element[]) => {
    const list = byShelf.get(shelf) ?? []
    list.push({ x, node })
    byShelf.set(shelf, list)
  }
  for (const can of layout.cans) place(can.shelf, can.x, [canNode(can, rowEnds.has(can.index))])
  for (const d of layout.dividers) place(d.shelf, d.x, dividerNodes(steel, d.x, d.z, d.height))

  const children: Element[] = [
    ...boxNodes('black-lacquer', [-120, RACK.depth - 80, -40], [carriage + 260, 40, RACK.rail]),
    ...boxNodes('black-lacquer', [-120, 20, -40], [carriage + 260, 40, RACK.rail]),
    ...boxNodes(
      steel,
      [-20, 0, -RACK.deck],
      [carriage + RACK.panel + 20, RACK.depth + 20, RACK.deck],
    ),
  ]
  const posts = (b: number) => {
    const bx = bayStart(b, W) - RACK.post
    return [
      ...boxNodes(steel, [bx, RACK.depth - RACK.post, 0], [RACK.post, RACK.post, tall]),
      ...boxNodes(steel, [bx, 0, 0], [RACK.post, RACK.post, tall]),
    ]
  }
  for (let b = 0; b < bays; b++) {
    children.push(...posts(b))
    const x0 = bayStart(b, W)
    for (let row = S - 1; row >= 0; row--) {
      const shelf = b * S + row
      const z = shelfTop(row, S)
      children.push(...boxNodes(steel, [x0, 0, z - RACK.board], [W, RACK.depth, RACK.board]))
      const items = (byShelf.get(shelf) ?? []).sort((a, c) => a.x - c.x)
      for (const { node } of items) children.push(...node)
      if (options.showTags)
        for (const h of layout.holders)
          if (h.shelf === shelf) children.push(...holderNodes(steel, h.x, h.end, h.z, h.label))
    }
    children.push(...boxNodes(steel, [x0, 0, tall - 10 - RACK.board], [W, RACK.depth, RACK.board]))
  }
  children.push(...posts(bays))
  if (options.crank) {
    children.push(
      ...boxNodes(steel, [carriage, 0, -RACK.deck], [RACK.panel, RACK.depth + 20, tall + 50]),
    )
    children.push(crankNode(carriage + RACK.panel, 1050))
  }
  children.push(s('g', { class: 'vault-lift' }))

  const materials = new Set<VaultMaterial>([steel, 'black-lacquer'])
  for (const can of layout.cans) materials.add(can.finish)
  // The camera's first frame starts under the cap board, at the top-left of the first bay.
  const [sx, sy] = project(bayStart(0, W) - RACK.post, 0, shelfTop(0, S) + RACK.pitch - RACK.board)
  return s(
    'svg',
    {
      class: 'vault-scene',
      viewBox: `${r1(minX - pad)} ${r1(minY - pad)} ${width} ${height}`,
      width,
      height,
      role: 'img',
      'aria-label': options.label,
      'data-start': `${r1(sx)} ${r1(sy)}`,
    },
    [s('style', vaultCss(materials)), vaultDefs(materials), ...children],
  )
}
