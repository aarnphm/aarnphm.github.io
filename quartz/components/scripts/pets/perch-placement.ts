import { trailHost, trailScene } from '../../../util/trail-scene'
import { circleClearance, type CollisionActor } from '../../visitor-trail/actor-spacing'
import { trailClockRegion } from '../../visitor-trail/clock-clearance'
import { visibleTextBoxes } from '../pet-text-boxes'
import {
  clamp,
  edgeY,
  height,
  margin,
  petState,
  viewBottom,
  viewTop,
  type Edge,
  type Kind,
  type Perch,
  type Point,
} from './actor-model'

const replaced = new Set(['IMG', 'VIDEO', 'CANVAS', 'IFRAME', 'PRE', 'TABLE', 'HR'])
const headings = new Set(['H1', 'H2', 'H3', 'H4'])
const perchTags =
  '#quartz-root :is(h1, h2, h3, h4, hr, pre, figure, img, video, table, blockquote, .callout), [data-pet-perch]'

// Is `el` part of the document's flow, scrolled by the window? Boxes inside fixed or sticky
// furniture, or inside their own scroller, move without the pet, so it never holds them.
export const inFlow = (el: Element) => {
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const style = getComputedStyle(node)
    if (style.position === 'fixed' || style.position === 'sticky') return false
    if (/(auto|scroll|hidden)/.test(style.overflowY + style.overflowX)) {
      if (node.scrollHeight > node.clientHeight + 1 || node.scrollWidth > node.clientWidth + 1)
        return false
    }
  }
  return true
}

const visibleLine = (style: CSSStyleDeclaration, side: 'Top' | 'Bottom') =>
  parseFloat(style[`border${side}Width`]) > 0 &&
  style[`border${side}Style`] !== 'none' &&
  !/rgba\(.*,\s*0\)$|transparent/.test(style[`border${side}Color`])

const filled = (style: CSSStyleDeclaration) =>
  !/rgba\(.*,\s*0\)$|transparent/.test(style.backgroundColor)

// The box the pet holds on `el`. A heading's box is its first line of text, so the pet stands on
// the letters rather than on the empty end of a full-width block.
export function boxOf(el: Element): DOMRect {
  if (!headings.has(el.tagName)) return el.getBoundingClientRect()
  const range = document.createRange()
  range.selectNodeContents(el)
  for (const rect of range.getClientRects()) if (rect.width > 4 && rect.height > 4) return rect
  return el.getBoundingClientRect()
}

// Edges in view near `anchor`. Boxes come from the element stacks under a grid of points across the
// viewport plus a short list of tags worth standing on, so the cost follows the view rather than
// the size of the page.
export function edgesNear(anchor: Point, avoid: Element | null): Edge[] {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const seen = new Set<Element>()
  for (let i = 0; i < 7; i++)
    for (let j = 0; j < 6; j++)
      for (const el of document.elementsFromPoint(vw * (0.06 + i * 0.147), vh * (0.08 + j * 0.168)))
        seen.add(el)
  for (const el of document.querySelectorAll(perchTags)) seen.add(el)

  const edges: Edge[] = []
  const minX = margin + petState.width / 2
  const maxX = vw - margin - petState.width / 2
  const h = height()
  for (const el of seen) {
    if (el === document.documentElement || el === document.body || petState.root?.contains(el))
      continue
    if (petState.closing?.contains(el)) continue
    const rect = boxOf(el)
    if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) continue
    const isRule = el.tagName === 'HR'
    const isText = headings.has(el.tagName)
    if (rect.width < petState.width * 0.8 || (!isRule && !isText && rect.height < 28)) continue
    if (el.closest('header, nav, dialog, .popover, section.header, [aria-hidden="true"]')) continue
    const style = getComputedStyle(el)
    if (style.visibility === 'hidden' || Number(style.opacity) === 0) continue
    const solid = replaced.has(el.tagName) || filled(style)
    const lineTop = visibleLine(style, 'Top')
    const lineBottom = visibleLine(style, 'Bottom')
    // Page-wide containers only count where they draw a line.
    if (rect.width > vw * 0.92 && !lineTop && !lineBottom) continue
    const top = isText || isRule || solid || lineTop
    const bottom = !isRule && !isText && (solid || lineBottom)
    const lo = clamp(rect.left + petState.width * 0.3, minX, maxX)
    const hi = clamp(rect.right - petState.width * 0.3, minX, maxX)
    if (hi < lo) continue
    // A sitter takes the corner nearer the anchor with its legs over the box's side, and a hanger
    // hangs off the corner by one hand, so neither dangles over what the box holds.
    const side = anchor.x > (rect.left + rect.right) / 2 ? 1 : -1
    const corner = (off: number) => {
      const out = side > 0 ? rect.right + off * petState.unit : rect.left - off * petState.unit
      if (out >= minX && out <= maxX) return out
      const other = side > 0 ? rect.left - off * petState.unit : rect.right + off * petState.unit
      return other >= minX && other <= maxX ? other : null
    }
    const push = (kind: Kind, y: number, weight: number) => {
      const x =
        kind === 'sit' ? corner(0.36) : kind === 'hang' ? corner(0.1) : clamp(anchor.x, lo, hi)
      if (x === null) return
      const t = (x - rect.left) / rect.width
      const y0 = y - (edgeY[kind] - viewTop) * petState.unit
      if (y0 < margin || y0 + h > vh - margin) return
      const score =
        Math.hypot(anchor.x - x, anchor.y - y) * weight +
        (el === avoid ? 400 : 0) +
        Math.random() * 60
      edges.push({ el, kind, t, at: { x, y }, score })
    }
    if (top) {
      const sit = !isRule && !isText && rect.height > 0.6 * petState.unit && Math.random() < 0.5
      push(sit ? 'sit' : 'stand', rect.top, 1)
    }
    if (bottom && rect.height > 40) push('hang', rect.bottom, 1.2)
  }
  return edges.sort((a, b) => a.score - b.score)
}

export const campSolids = (): CollisionActor[] => {
  const obstacles: CollisionActor[] = []
  const scene = trailScene(document)
  if (!scene || trailHost(scene)?.hidden) return obstacles
  for (const element of scene.querySelectorAll<SVGGElement>('[data-camp-solid]')) {
    const matrix = element.getScreenCTM()
    const parent = element.parentElement
    const ground = parent instanceof SVGGraphicsElement ? parent.getScreenCTM() : null
    const radius = Number(element.dataset.solidRadius)
    const id = element.dataset.campSolid
    if (!matrix || !ground || !id || !Number.isFinite(radius) || radius <= 0) continue
    obstacles.push({
      id,
      kind: 'trunk',
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: radius * Math.hypot(ground.a, ground.b),
    })
  }
  return obstacles
}
// Controls in view, so the pet never stands over a button. Prose links are already text boxes; an
// icon-only link has no text of its own and joins the controls. The trail scene's own targets (a
// watering button on every tree) stay walkable.
const controls = 'button, [role="button"], input, select, textarea, summary, a[href]'
const controlBoxes = () => {
  const boxes: DOMRect[] = []
  const scene = trailScene(document)
  for (const el of document.querySelectorAll<HTMLElement>(controls)) {
    if (petState.root?.contains(el) || scene?.contains(el)) continue
    if (el.tagName === 'A' && el.textContent?.trim()) continue
    const rect = el.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) continue
    if (rect.bottom < 0 || rect.top > window.innerHeight) continue
    if (rect.right < 0 || rect.left > window.innerWidth) continue
    boxes.push(rect)
  }
  return boxes
}

export const petObstacles = () => ({
  text: visibleTextBoxes(document, petState.root),
  panels: [
    ...Array.from(document.querySelectorAll<HTMLElement>('.homepage-board'), panel =>
      panel.getBoundingClientRect(),
    ),
    ...[trailClockRegion(document)].flatMap(region => (region ? [region] : [])),
    ...controlBoxes(),
  ],
  camps: campSolids(),
})
export type PetObstacles = ReturnType<typeof petObstacles>
export type PlacementMeasurements = { obstacles?: PetObstacles }

// A snapshot lasts only for the synchronous search, so scrolling and DOM changes get fresh boxes.
export const coversText = (edge: Edge, obstacles = petObstacles()) => {
  const scale = edge.scale ?? 1
  const left = edge.at.x - (petState.width * scale) / 2 - 2
  const right = edge.at.x + (petState.width * scale) / 2 + 2
  const top = edge.at.y - (edgeY[edge.kind] - viewTop) * petState.unit * scale - 2
  const bottom = top + height(scale) + 4
  const intersects = (rect: { left: number; right: number; top: number; bottom: number }) =>
    rect.left < right && rect.right > left && rect.top < bottom && rect.bottom > top
  const foot = { x: edge.at.x + window.scrollX, y: edge.at.y + window.scrollY }
  return (
    obstacles.panels.some(intersects) ||
    obstacles.camps.some(
      other => circleClearance(foot, petState.width * 0.35 * scale, other) < 0,
    ) ||
    obstacles.text.some(intersects)
  )
}

// The floor, too, keeps off text: a client point where the pet could stand on its own line.
export const floorStep = 16
const floorY = () => window.innerHeight - margin - (viewBottom - edgeY.floor) * petState.unit
export const floorClear = (x: number, y = floorY(), obstacles = petObstacles()) =>
  !coversText({ el: null, kind: 'floor', t: 0, at: { x, y }, score: 0 }, obstacles)

// Scan out both ways, then move up when a line of text blocks the whole floor.
export const floorNear = (
  x: number,
  fits?: (p: Perch) => boolean,
  seen: PlacementMeasurements = {},
): Perch | null => {
  const obstacles = (seen.obstacles ??= petObstacles())
  const lo = margin + petState.width / 2
  const hi = window.innerWidth - margin - petState.width / 2
  const start = clamp(x, lo, hi)
  const bottom = floorY()
  const top = margin + (edgeY.floor - viewTop) * petState.unit
  for (let y = bottom; y >= top; y -= floorStep)
    for (let k = 0; k * floorStep <= hi - lo; k++)
      for (const c of k ? [start + k * floorStep, start - k * floorStep] : [start]) {
        if (c < lo || c > hi || !floorClear(c, y, obstacles)) continue
        const p: Perch = {
          el: null,
          kind: 'floor',
          t: c / window.innerWidth,
          foot: y === bottom ? undefined : { x: c + window.scrollX, y: y + window.scrollY },
        }
        if (!fits || fits(p)) return p
      }
  return null
}

// How far along the floor from `from` toward `to` the pet can walk with text clear underfoot.
export const floorReach = (from: number, to: number) => {
  const obstacles = petObstacles()
  const dir = Math.sign(to - from)
  let x = from
  while (Math.abs(to - x) > floorStep && floorClear(x + dir * floorStep, floorY(), obstacles))
    x += dir * floorStep
  return Math.abs(to - x) <= floorStep && floorClear(to, floorY(), obstacles) ? to : x
}

// The first edge that is actually in view at the point the pet would hold, moves with the page,
// and leaves the text around it clear.
export function pickEdge(
  anchor: Point,
  avoid: Element | null,
  fits?: (p: Perch) => boolean,
  seen: PlacementMeasurements = {},
): Edge | null {
  const obstacles = (seen.obstacles ??= petObstacles())
  for (const edge of edgesNear(anchor, avoid).slice(0, 20)) {
    const probe = edge.kind === 'hang' ? edge.at.y - 3 : edge.at.y + 3
    const hit = document.elementsFromPoint(edge.at.x, probe).find(e => !petState.root?.contains(e))
    if (edge.kind === 'stand' && (!hit || !(edge.el === hit || edge.el!.contains(hit)))) continue
    if (!inFlow(edge.el!) || coversText(edge, obstacles)) continue
    if (fits && !fits(edge)) continue
    return edge
  }
  return null
}

// The document position (top-left of the pet) that puts its held edge on `perch`.
export function placeOn(p: Perch): Point {
  if (p.foot) {
    const scale = p.scale ?? 1
    return {
      x: clamp(
        p.foot.x - (petState.width * scale) / 2,
        window.scrollX + margin,
        window.scrollX + window.innerWidth - margin - petState.width * scale,
      ),
      y: p.foot.y - (edgeY[p.kind] - viewTop) * petState.unit * scale,
    }
  }
  let x: number
  let y: number
  if (p.el) {
    const rect = boxOf(p.el)
    x = rect.left + p.t * rect.width
    y = (p.kind === 'hang' ? rect.bottom : rect.top) + (p.drop ?? 0)
  } else {
    x = p.t * window.innerWidth
    y = window.innerHeight - margin - (viewBottom - edgeY.floor) * petState.unit
  }
  const scale = p.scale ?? 1
  x = clamp(
    x,
    margin + (petState.width * scale) / 2,
    window.innerWidth - margin - (petState.width * scale) / 2,
  )
  return {
    x: x - (petState.width * scale) / 2 + window.scrollX,
    y: y - (edgeY[p.kind] - viewTop) * petState.unit * scale + window.scrollY,
  }
}
