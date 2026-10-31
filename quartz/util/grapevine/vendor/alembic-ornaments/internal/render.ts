import type { Composition, Part, Rect } from './presets.js'
import { orient, polyToPath, simplify, type Matrix } from './geometry.js'

const fmtMatrix = (m: Matrix) =>
  [m.a, m.b, m.c, m.d, m.e, m.f].map(v => String(Math.round(v * 1000) / 1000)).join(' ')

const isIdentity = (m: Matrix) =>
  m.a === 1 && m.b === 0 && m.c === 0 && m.d === 1 && m.e === 0 && m.f === 0

function partD(part: Part, tol: number): string {
  let d = ''
  for (const s of part.shapes) {
    const poly = simplify(s.poly, tol)
    if (poly.length >= 3) d += polyToPath(orient(poly, s.hole))
  }
  return d
}

function hashKey(input: string): string {
  let h = 2166136261
  for (let i = 0; i < input.length; i++) h = Math.imul(h ^ input.charCodeAt(i), 16777619)
  return (h >>> 0).toString(36)
}

const r1 = (v: number) => Math.round(v * 10) / 10

/**
 * SVG in `currentColor`, so an ornament prints in whatever ink the text
 * around it uses. Parts placed more than once go in <defs> and are <use>d.
 * A block ornament is a solid rect with the drawing masked out of it, so
 * the paper shows through without the SVG knowing what colour paper is.
 */
/** `tol` is the outline simplification in the composition's units (0.12 upstream). */
export function render(
  c: Composition,
  key: string,
  tol = 0.12,
): { svg: string; width: number; height: number } {
  const id = `orn-${hashKey(key)}`
  const uses = new Map<number, number>()
  for (const pl of c.placements) uses.set(pl.part, (uses.get(pl.part) ?? 0) + 1)

  const defs: string[] = []
  const ds = new Map<number, string>()
  const dOf = (i: number) => {
    if (!ds.has(i)) ds.set(i, partD(c.parts[i], tol))
    return ds.get(i)!
  }
  for (const [i, n] of uses) if (n > 1) defs.push(`<path id="${id}-${i}" d="${dOf(i)}"/>`)

  const use = (pl: { part: number; m: Matrix }) => {
    const transform = isIdentity(pl.m) ? '' : ` transform="matrix(${fmtMatrix(pl.m)})"`
    return (uses.get(pl.part) ?? 0) > 1
      ? `<use href="#${id}-${pl.part}"${transform}/>`
      : `<path d="${dOf(pl.part)}"${transform}/>`
  }
  const clipPath = (key: string, r: Rect) => {
    defs.push(
      `<clipPath id="${id}-${key}"><rect x="${r1(r.x)}" y="${r1(r.y)}" width="${r1(r.w)}" height="${r1(r.h)}"/></clipPath>`,
    )
    return ` clip-path="url(#${id}-${key})"`
  }

  let loose = c.placements
    .filter(pl => pl.group === undefined)
    .map(use)
    .join('')
  if (c.clip && loose) loose = `<g${clipPath('clip', c.clip)}>${loose}</g>`
  const grouped = (c.groups ?? []).map((g, k) => {
    const inner = c.placements
      .filter(pl => pl.group === k)
      .map(use)
      .join('')
    if (!inner) return ''
    const transform = isIdentity(g.m) ? '' : ` transform="matrix(${fmtMatrix(g.m)})"`
    return `<g${transform}${g.clip ? clipPath(`g${k}`, g.clip) : ''}>${inner}</g>`
  })
  let ink = loose + grouped.join('')

  if (c.block) {
    const { outer: o, inner } = c.block
    const rect = `x="${r1(o.x)}" y="${r1(o.y)}" width="${r1(o.w)}" height="${r1(o.h)}"`
    const rectD = (r: Rect) => `M${r1(r.x)} ${r1(r.y)}h${r1(r.w)}v${r1(r.h)}h${r1(-r.w)}Z`
    defs.push(
      `<mask id="${id}-cut" maskUnits="userSpaceOnUse" ${rect}><rect ${rect} fill="#fff"/><g fill="#000">${ink}</g></mask>`,
    )
    ink = inner
      ? `<path fill-rule="evenodd" d="${rectD(o)}${rectD(inner)}" mask="url(#${id}-cut)"/>`
      : `<rect ${rect} mask="url(#${id}-cut)"/>`
  }

  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${r1(c.width)} ${r1(c.height)}" ` +
    `width="${r1(c.width)}" height="${r1(c.height)}" fill="currentColor">` +
    (defs.length ? `<defs>${defs.join('')}</defs>` : '') +
    ink +
    `</svg>`
  return { svg, width: c.width, height: c.height }
}
