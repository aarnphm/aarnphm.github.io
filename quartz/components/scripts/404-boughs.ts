// The trees at either edge of the view shed what the season has put on them: leaves in autumn and
// blossom in spring, turning over as they fall and carried off on the wind. The left-hand tree has
// people in it: someone sitting out on a limb with their legs swinging, and an arborist on a rope
// from the limb above, whose work goes with the season. They saw out deadwood in spring and summer,
// shake the leaves down in autumn, and string lights along the limb in winter.

import type { Voices } from './404-crew'
import { type Grid, type Pt, type RGB, BAYER, col, dot, hash, line, row, stamp } from './404-pixel'

type Limb = { x0: number; y0: number; x1: number; y1: number; w0: number; w1: number }
type Crown = { segs: Limb[]; tips: [number, number, number][] }

// In scene units: the seat, the trunk end of the limb the lights run up and its far end, and the
// point on a limb above that the rope is tied to.
export type Bough = { seat: Pt; root: Pt; end: Pt; anchor: Pt }
export type Work = 'saw' | 'shake' | 'lights'

type Leaf = { x: number; y: number; vx: number; vy: number; spin: number; tone: number }
export type Shed = { leaves: Leaf[]; spawn: number }

export type BoughInk = {
  skin: RGB
  hat: RGB
  coats: [RGB, RGB]
  trousers: RGB
  boots: RGB
  harness: RGB
  paper: RGB
  apple: RGB
  steel: RGB
  rope: RGB
  bark: RGB
  dust: RGB
  lights: RGB[]
  glow: RGB
  steam: RGB
  dark: boolean
}

// Sprites face right. Sitting: the thighs lie along the limb, the shins hang down in front of it.
// P is whatever they have in hand.
const PERCHED = ['.HH.', 'HHHH', '.SS.', '.BBP', '.BDD', '...D', '...D', '...K']
const SWUNG = ['.HH.', 'HHHH', '.SS.', '.BBP', '.BDD', '...D', '..D.', '..K.']
// Hanging: one hand up on the rope over the helmet, the harness at the hips.
const HANGING = ['.S..', '.S..', '.HH.', 'HHHH', '.SS.', 'BBBB', 'BGGB', '.DD.', '.DD.', '.KK.']

// The perch is the most nearly level limb thick enough to sit on, clear of the board and far enough
// inside the view that someone on it is seen. A tree grown steep has no such limb, and then they sit
// in its lowest fork that is seen, and the lights run up whichever limb leaves the fork furthest
// out. The rope hangs from the limb that best clears the seat above and out toward the gate, far
// enough out that it drops past the limbs fanning up from the seat.
export function findBough(
  tree: Crown,
  g: Grid,
  view: [number, number, number, number],
  hidden: (x: number, y: number) => boolean,
): Bough | null {
  const px = g.px
  const seen = (x: number, y: number) =>
    x > view[0] + 5 * px &&
    x < view[2] - 5 * px &&
    y > view[1] + 14 * px &&
    y < view[3] - 20 * px &&
    !hidden(x, y)
  const children = (p: Limb) => tree.segs.filter(s => s !== p && s.x0 === p.x1 && s.y0 === p.y1)
  let seat: Pt | null = null
  let root: Pt = [0, 0]
  let end: Pt = [0, 0]
  let perch: Limb | null = null
  let best = -Infinity
  for (const s of tree.segs) {
    const dx = s.x1 - s.x0
    const dy = s.y1 - s.y0
    const len = Math.hypot(dx, dy)
    if (s.w0 < 2 * px || s.w0 > 8 * px || Math.abs(dy) > Math.abs(dx) * 1.2 || len < 5 * px)
      continue
    if (!seen((s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2 - 8 * px) || !seen(s.x1, s.y1)) continue
    const score = len / px + s.w0 / px - (Math.abs(dy) / Math.abs(dx)) * 8
    if (score > best) {
      best = score
      perch = s
    }
  }
  if (perch) {
    seat = [(perch.x0 + perch.x1) / 2, (perch.y0 + perch.y1) / 2 - (perch.w0 + perch.w1) / 4]
    root = [perch.x0, perch.y0]
    end = [perch.x1, perch.y1 - perch.w1 / 2]
  } else {
    const trunk = tree.segs[0]
    const out = trunk && trunk.x0 < (view[0] + view[2]) / 2 ? 1 : -1
    let fork: Limb | null = null
    for (const s of tree.segs) {
      if (s.w1 < 2 * px || children(s).length < 2 || !seen(s.x1, s.y1 - 8 * px)) continue
      if (!fork || s.y1 > fork.y1) fork = s
    }
    if (!fork) return null
    seat = [fork.x1, fork.y1 - fork.w1 / 4]
    root = seat
    // Up the limb leaving the fork furthest out, while it stays thick enough to hang lights on.
    let limb = fork
    for (let n = 0; n < 4; n++) {
      const next = children(limb).reduce<Limb | null>(
        (a, c) => (!a || (c.x1 - c.x0) * out > (a.x1 - a.x0) * out ? c : a),
        null,
      )
      if (!next || next.w1 < 1.2 * px || !seen(next.x1, next.y1)) break
      limb = next
    }
    end = [limb.x1, limb.y1 - limb.w1 / 2]
    if (limb === fork) end = [seat[0] + out * 16 * px, seat[1] - 16 * px]
  }
  const out = Math.sign(end[0] - root[0]) || 1
  let anchor: Pt = [seat[0] + out * 18 * px, seat[1] - 28 * px]
  best = -Infinity
  for (const s of tree.segs) {
    if (s.w1 < 1.2 * px || s === perch) continue
    for (const [x, y] of [
      [s.x1, s.y1],
      [(s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2],
    ]) {
      const ox = (x - seat[0]) * out
      const oy = seat[1] - y
      if (ox < 7 * px || ox > 26 * px || oy < 16 * px || oy > 48 * px || !seen(x, y)) continue
      const score = -Math.abs(ox - 18 * px) - Math.abs(oy - 28 * px) * 0.5
      if (score > best) {
        best = score
        anchor = [x, y]
      }
    }
  }
  return { seat, root, end, anchor }
}

export function workFor(s: { autumn: number; bare: number; ground: number }): Work {
  if (s.bare > 0.45 || s.ground > 0.3) return 'lights'
  return s.autumn > 0.35 ? 'shake' : 'saw'
}

// Leaves fall at about a metre and a half a second, a tree being some nine hundred units tall, and
// blossom at half that; both tumble, so they drop fastest edge-on and slide sideways as they turn.
export function stepShed(
  shed: Shed,
  crowns: Crown[],
  env: {
    leaves: number
    blossom: number
    wind: number
    shake: Pt | null
    box: [number, number, number, number]
  },
  dt: number,
) {
  const shaken = env.shake ? 5 : 0
  const rate = (4 * env.leaves + 3 * env.blossom) * (0.5 + env.wind) + shaken
  const tips = crowns.flatMap(c => c.tips)
  if (tips.length) shed.spawn += rate * dt
  while (shed.spawn >= 1 && shed.leaves.length < 80) {
    shed.spawn -= 1
    let [x, y] = tips[Math.floor(Math.random() * tips.length)]
    // The arborist's share comes off the limbs round the rope.
    if (env.shake && Math.random() * rate < shaken) {
      const [ax, ay] = env.shake
      let near = Infinity
      for (let k = 0; k < 12; k++) {
        const t = tips[Math.floor(Math.random() * tips.length)]
        const dist = Math.hypot(t[0] - ax, t[1] - ay)
        if (dist < near) [near, x, y] = [dist, t[0], t[1]]
      }
    }
    const petal = Math.random() * (env.leaves + env.blossom) < env.blossom
    shed.leaves.push({
      x,
      y,
      vx: (8 + 70 * env.wind) * (0.6 + 0.8 * Math.random()),
      vy: (petal ? 28 : 55) + 30 * Math.random(),
      spin: Math.random() * Math.PI * 2,
      tone: petal ? -1 : Math.floor(Math.random() * 4),
    })
  }
  shed.spawn = Math.min(shed.spawn, 1)
  for (const l of shed.leaves) {
    l.spin += dt * (2.4 + (l.tone & 1) * 1.1)
    l.x += (l.vx + 22 * Math.sin(l.spin * 1.3)) * dt
    l.y += l.vy * (0.45 + 0.55 * Math.abs(Math.cos(l.spin))) * dt
  }
  const [x0, y0, x1, y1] = env.box
  shed.leaves = shed.leaves.filter(l => l.y < y1 && l.y > y0 - 50 && l.x < x1 + 20 && l.x > x0 - 20)
}

export function paintShed(d: Uint8ClampedArray, g: Grid, shed: Shed, leaf: RGB[], petal: RGB) {
  for (const l of shed.leaves) {
    const i = col(g, l.x)
    const j = row(g, l.y)
    const c = l.tone < 0 ? petal : leaf[l.tone]
    dot(d, g, i, j, c)
    // Broadside a leaf is two pixels across, edge-on one.
    if (Math.cos(l.spin) > 0.2) dot(d, g, i + (Math.sin(l.spin) > 0 ? 1 : -1), j, c)
  }
}

export function paintBough(
  d: Uint8ClampedArray,
  g: Grid,
  b: Bough,
  work: Work,
  ink: BoughInk,
  clock: number,
  heads: Voices['heads'],
) {
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  const glow = (i: number, j: number, reach: number, alpha: number) => {
    for (let dj = -reach; dj <= reach; dj++)
      for (let di = -reach; di <= reach; di++) {
        const r = Math.hypot(di, dj)
        if (r > 0) plot(i + di, j + dj, ink.glow, (1 - r / (reach + 0.8)) * alpha)
      }
  }
  const out = b.end[0] >= b.root[0]
  const si = col(g, b.seat[0])
  const sj = row(g, b.seat[1])

  // Winter's lights go up along the underside of the perch, from the trunk side out to its end, and
  // twinkle out of step with each other.
  const [ei, ej] = [col(g, b.end[0]), row(g, b.end[1])]
  if (work === 'lights') {
    const [ri, rj] = [col(g, b.root[0]), row(g, b.root[1])]
    const span = Math.max(1, Math.round(Math.hypot(ei - ri, ej - rj)))
    for (let k = 0; k <= span; k += 2) {
      const u = k / span
      const i = Math.round(ri + (ei - ri) * u)
      const j = Math.round(rj + (ej - rj) * u) + 2 + ((k >> 1) & 1)
      const lit = hash(k, Math.floor(clock * 1.6 + k * 0.37), 61) > 0.3
      const c = ink.lights[(k >> 1) % ink.lights.length]
      if (lit && ink.dark) glow(i, j, 1, 0.45)
      dot(d, g, i, j, lit ? c : ink.bark)
    }
  }

  // The sitter, swinging their legs, with a book in spring and summer, an apple in autumn, and a
  // steaming mug in winter; at night a lantern hangs on the limb beside them.
  const swing = Math.floor(clock / 0.55) % 2 === 1
  const prop = work === 'shake' ? ink.apple : ink.paper
  const sitCoat = ink.coats[0]
  const top = sj - 5
  const i0 = out ? si - 1 : si - 2
  stamp(d, g, swing ? SWUNG : PERCHED, i0, top, !out, ch =>
    ch === 'H'
      ? ink.hat
      : ch === 'S'
        ? ink.skin
        : ch === 'B'
          ? sitCoat
          : ch === 'D'
            ? ink.trousers
            : ch === 'K'
              ? ink.boots
              : ch === 'P'
                ? prop
                : null,
  )
  heads.sitter = [i0 + (out ? 1 : 2), top]
  const hand = out ? i0 + 3 : i0
  if (work === 'lights') {
    // A scarf, and steam off the mug.
    dot(d, g, i0 + 1, top + 3, ink.apple)
    dot(d, g, i0 + 2, top + 3, ink.apple)
    if (Math.floor(clock / 0.4) % 3)
      plot(hand, top + 2 - (Math.floor(clock / 0.4) % 2), ink.steam, 0.8)
  }
  if (ink.dark) {
    const li = si + (out ? -3 : 3)
    dot(d, g, li, sj + 1, ink.rope)
    glow(li, sj + 2, 2, 0.5 + 0.15 * hash(Math.floor(clock * 5), 1, 43))
    dot(d, g, li, sj + 2, ink.lights[0])
  }

  // The arborist on the rope, swaying a little, and bouncing on it while they shake the limb.
  const [ax, ay] = b.anchor
  const sway = Math.sin(clock * 1.2) * 0.06
  const bounce = work === 'shake' ? Math.abs(Math.sin(clock * 7)) * g.px : 0
  const len = b.seat[1] - ay - 6 * g.px + bounce
  const hx = ax + Math.sin(sway) * len
  const hy = ay + Math.cos(sway) * len
  const ai = col(g, ax)
  const aj = row(g, ay)
  const hi = col(g, hx)
  const hj = row(g, hy)
  line(d, g, ai, aj, hi, hj, ink.rope)
  // A turn of rope round the limb where it is tied.
  dot(d, g, ai - 1, aj, ink.rope)
  dot(d, g, ai + 1, aj, ink.rope)
  const ri = hi - 1
  stamp(d, g, HANGING, ri, hj, false, ch =>
    ch === 'H'
      ? ink.harness
      : ch === 'S'
        ? ink.skin
        : ch === 'B'
          ? ink.coats[1]
          : ch === 'G'
            ? ink.harness
            : ch === 'D'
              ? ink.trousers
              : ch === 'K'
                ? ink.boots
                : null,
  )
  heads.arborist = [ri + 1, hj + 2]
  // The rope's tail hangs on below them.
  for (let k = 10; k <= 13; k++) dot(d, g, ri + 1, hj + k, ink.rope)
  const wi = ri + 4
  const wj = hj + 5
  if (work === 'saw') {
    // A dead stub off the limb above, and the saw going at it; the dust drifts down under the cut.
    line(d, g, ri + 7, hj - 2, ri + 6, hj + 9, ink.bark)
    const stroke = Math.round(Math.sin(clock * 10))
    dot(d, g, wi, wj, ink.skin)
    for (let k = 1; k <= 3; k++) dot(d, g, wi + k + stroke, wj, ink.steel)
    for (let k = 0; k < 4; k++) {
      const fall = (clock * 9 + k * 2.7) % 9
      dot(
        d,
        g,
        ri + 6 + Math.round(hash(k, Math.floor(clock * 9 + k * 2.7) >> 3, 7) * 2 - 1),
        wj + 1 + Math.floor(fall),
        ink.dust,
      )
    }
  } else if (work === 'shake') {
    // Both hands up on the rope, working it.
    dot(d, g, ri + 2, hj + 1, ink.skin)
  } else {
    // The end of the string of lights, fed out from the hand to the end of the limb.
    dot(d, g, wi, wj, ink.skin)
    line(d, g, wi, wj, ei, ej + 2, ink.bark)
  }
}
