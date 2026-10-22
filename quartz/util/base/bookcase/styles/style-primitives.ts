import { box, COS30, fitText, project, type Segment } from '../geometry'
import {
  BOARD,
  type Orientation,
  type Plan,
  type Plate,
  type ShelfMaterial,
  type ShelfOptions,
  type ShelfPlacement,
  type StyleContext,
  type StyleHalo,
  type StyleLayer,
  type StylePart,
  type StyleSolid,
  type Tone,
  type Tones,
  type World,
} from './style-model'
import { at, chain, seg, toWorld } from './world-space'

export function solidOf(part: StylePart, material: ShelfMaterial): StyleSolid {
  return { part, material, faces: [], lines: [], creases: [], outline: null, edges: [], screws: [] }
}

export function boxPart(
  part: StylePart,
  material: ShelfMaterial,
  [x, y, z]: World,
  [w, d, h]: World,
  tones: Tones = {},
): StyleSolid {
  const s = box(x, y, z, w, d, h)
  const solid = solidOf(part, material)
  solid.faces = [
    { points: s.front, tone: tones.front ?? 'face', o: 'front', tall: h > w },
    { points: s.side, tone: tones.side ?? 's2', o: 'side', tall: h > d },
    { points: s.top, tone: tones.top ?? 'top', o: 'top', tall: d > w },
  ]
  if (material === 'birch-ply' && h <= 16) {
    const stripes = [1 / 3, 2 / 3].flatMap(k =>
      chain([
        [x, y, z + h * k],
        [x + w, y, z + h * k],
        [x + w, y + d, z + h * k],
      ]),
    )
    solid.lines.push({ kind: 'ply', segments: stripes })
  }
  if (material === 'steel') {
    const tall = h >= w && h >= d
    solid.lines.push({
      kind: 'highlight',
      segments: [
        tall
          ? seg([x + 2, y, z + 2], [x + 2, y, z + h - 2])
          : seg([x + 2, y, z + h - 2], [x + w - 2, y, z + h - 2]),
      ],
    })
  }
  solid.creases = s.creases
  solid.outline = s.edge
  return solid
}

/**
 * Right prism over an xy polygon (L-shaped boards and crowns). Side faces whose outward normal
 * points to +x or -y are visible; a corner between two visible faces gets a crease, a corner
 * between a visible and a hidden face an edge.
 */
export function prismPart(
  part: StylePart,
  material: ShelfMaterial,
  plan: Plan,
  z0: number,
  h: number,
  tones: Tones = {},
): StyleSolid {
  let area = 0
  for (const [i, [ax, ay]] of plan.entries()) {
    const [bx, by] = plan[(i + 1) % plan.length]
    area += ax * by - bx * ay
  }
  const q = area < 0 ? [...plan].reverse() : [...plan]
  const n = q.length
  const z1 = z0 + h
  const next = (i: number) => q[(i + 1) % n]
  const visible = q.map(([ax, ay], i) => {
    const [bx, by] = next(i)
    return by - ay + (bx - ax) > 0
  })
  const solid = solidOf(part, material)
  for (const [i, [ax, ay]] of q.entries()) {
    if (!visible[i]) continue
    const [bx, by] = next(i)
    const front = Math.abs(bx - ax) > Math.abs(by - ay)
    solid.faces.push({
      points: [project(ax, ay, z0), project(bx, by, z0), project(bx, by, z1), project(ax, ay, z1)],
      tone: front ? (tones.front ?? 'face') : (tones.side ?? 's2'),
      o: front ? 'front' : 'side',
      tall: h > Math.hypot(bx - ax, by - ay),
    })
  }
  solid.faces.push({
    points: q.map(([x, y]) => project(x, y, z1)),
    tone: tones.top ?? 'top',
    o: 'top',
  })
  for (const [i, [x, y]] of q.entries()) {
    const before = visible[(i - 1 + n) % n]
    const [nx, ny] = next(i)
    const riser = seg([x, y, z0], [x, y, z1])
    if (before && visible[i]) solid.creases.push(riser)
    else if (before !== visible[i]) solid.edges.push(riser)
    if (visible[i]) {
      solid.creases.push(seg([x, y, z1], [nx, ny, z1]))
      solid.edges.push(seg([x, y, z0], [nx, ny, z0]))
    } else solid.edges.push(seg([x, y, z1], [nx, ny, z1]))
  }
  return solid
}

export function panelPart(
  part: StylePart,
  material: ShelfMaterial,
  points: readonly World[],
  tone: Tone,
  outline = false,
): StyleSolid {
  const solid = solidOf(part, material)
  const span = (i: 0 | 1 | 2): [number, number] => {
    const values = points.map(p => p[i])
    return [Math.min(...values), Math.max(...values)]
  }
  const [xs, ys, zs] = [span(0), span(1), span(2)]
  const o: Orientation = ys[0] === ys[1] ? 'front' : xs[0] === xs[1] ? 'side' : 'top'
  const length = o === 'front' ? xs[1] - xs[0] : ys[1] - ys[0]
  const tall = o === 'top' ? ys[1] - ys[0] > xs[1] - xs[0] : zs[1] - zs[0] > length
  solid.faces = [{ points: points.map(at), tone, o, tall }]
  if (outline) solid.outline = points.map(at)
  return solid
}

export const withLines = (
  solid: StyleSolid,
  kind: StyleSolid['lines'][number]['kind'],
  segments: Segment[],
): StyleSolid => ({ ...solid, lines: [...solid.lines, { kind, segments }] })

export const halo = (
  kind: StyleHalo['kind'],
  material: ShelfMaterial,
  points: readonly World[],
): StyleHalo => ({ kind, material, points: points.map(at) })

export const solids = (parts: StyleSolid[], halos: StyleHalo[] = []): StyleLayer => ({
  kind: 'solids',
  halos,
  solids: parts,
})

/** Advance of one plate character: 5.4px Berkeley Mono at 0.6em plus 0.9px tracking. */
const PLATE_CHAR = 4.2
/** Plate room outside its label: 5 before the text, 13 after it for the ticks. */
const PLATE_PAD = 18

export function shelfLayer(
  ctx: StyleContext,
  p: ShelfPlacement,
  options: ShelfOptions,
): StyleLayer {
  const shelf = ctx.layout.shelves[p.shelf]
  const depth = ctx.layout.depth
  const w = (lx: number, ly: number): World => toWorld(p, lx, ly, 0)
  // The contact shadow lies behind the books, not at the back of a deeper board. It stops short of
  // the last book by its blur, so none of it shows on the empty board past the run.
  const end = Math.max(2, shelf.end - 6)
  const contact = options.contact ?? [
    w(2, depth - 14),
    w(end, depth - 14),
    w(end, depth),
    w(2, depth),
  ]
  let plate: Plate | null = null
  if (shelf.label && ctx.showTags) {
    const [x0, end] = options.board
    const full = shelf.label.toUpperCase()
    const x1 = Math.min(end, x0 + PLATE_CHAR * full.length + PLATE_PAD)
    // One unit of slack keeps a label that fits exactly from losing a character to rounding.
    const label = fitText(full, x1 - x0 - PLATE_PAD + 1, PLATE_CHAR, 1)
    const [z0, z1] = options.plateZ ?? [-BOARD + 1.5, -1.5]
    const q = (lx: number, lz: number) => at(toWorld(p, lx, -1.2, lz))
    const [e, f] = q(x0 + 5, (z0 + z1) / 2)
    if (label) {
      plate = {
        face: [q(x0, z0), q(x1, z0), q(x1, z1), q(x0, z1)],
        ticks: [0, 1, 2, 3].map(
          i => [q(x1 - 4 - i * 2, z0 + 1.5), q(x1 - 4 - i * 2, z1 - 1.5)] as Segment,
        ),
        matrix: [COS30, p.rot === 90 ? -0.5 : 0.5, 0, 1, e, f],
        label,
      }
    }
  }
  // A rot-90 run is painted from its far end to the near end; a stack from the bottom up.
  const books = p.rot === 90 ? [...ctx.runs[p.shelf]].reverse() : ctx.runs[p.shelf]
  return {
    kind: 'shelf',
    shelf: p.shelf,
    plate,
    contact: halo('contact', options.material, contact),
    books,
  }
}

export const standing = (
  shelf: number,
  origin: World,
  extra: Partial<ShelfPlacement> = {},
): ShelfPlacement => ({ shelf, origin, rot: 0, pose: 'standing', lane: 0, ...extra })

/** Bottom run first: the order in which stacked tiers are painted. */
export const bottomUp = (count: number) => Array.from({ length: count }, (_, i) => count - 1 - i)
