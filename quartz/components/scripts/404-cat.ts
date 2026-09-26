// The rubble on the right bank was a cat all along: a loaf of the ruin's stone facing the gate, a
// crack down its flank and moss on its back. It sleeps. A pointer nearby makes it peek with one
// eye; the open portal wakes it with round pupils and a puffed tail. Awake, it blinks, and it
// twitches an ear on every impulse of the clock. Now and then the tip of its tail flicks.
// Each part is its own blob, painted back to front, so the head, the tail and the paws are
// outlined where they lie over the body.

import {
  type Grid,
  type Pt,
  type RGB,
  type Tones,
  col,
  dot,
  hash,
  interior,
  line,
  noise,
  paintBlob,
  put,
  row,
  segment,
  smin,
  triangle,
} from './404-pixel'

export type Cat = {
  x: number
  y: number
  // 0 asleep, 1 peeking, 2 awake
  alert: number
  near: number
  calm: number
  blink: number
  shut: number
  twitch: number
  flick: number
  nextFlick: number
  puff: number
  look: number
}

export type CatInk = { stone: Tones; moss: Tones; eye: RGB; nose: RGB; ear: RGB; flowers: RGB[] }

// Where the chest meets the ground, and the head relative to it.
const HEAD: Pt = [-62, -50]

export function makeCat(x: number, ground: number): Cat {
  return {
    x,
    y: ground + 2,
    alert: 0,
    near: 0,
    calm: 0,
    blink: 3,
    shut: 0,
    twitch: 0,
    flick: 0,
    nextFlick: 5,
    puff: 0,
    look: 0,
  }
}

export const catHead = (c: Cat): Pt => [c.x + HEAD[0], c.y + HEAD[1]]

const NEAR = 260
const near = (c: Cat, pointer: { x: number; y: number }) => {
  const [hx, hy] = catHead(c)
  return pointer.x > -1e3 && Math.hypot(pointer.x - hx, pointer.y - hy) < NEAR
}
// Eyes follow the pointer across the head; with the portal open they fix on the gate.
function gaze(c: Cat, pointer: { x: number; y: number }, open: boolean) {
  const dx = pointer.x - catHead(c)[0]
  c.look = open ? -1 : pointer.x > -1e3 && Math.abs(dx) > 30 ? Math.sign(dx) : 0
}

// Without motion the cat goes straight to the pose it would settle into: peeking while the pointer
// is near, awake with round pupils and a puffed tail while the portal is open. No blinks, flicks or
// twitches.
export function poseCat(c: Cat, pointer: { x: number; y: number }, open: boolean) {
  c.alert = open ? 2 : near(c, pointer) ? 1 : 0
  c.puff = open ? 1 : 0
  c.shut = c.flick = c.twitch = 0
  gaze(c, pointer, open)
}

export function stepCat(
  c: Cat,
  dt: number,
  pointer: { x: number; y: number },
  heat: number,
  impulse: boolean,
) {
  c.near = near(c, pointer) ? c.near + dt : 0
  const target = heat > 0.3 ? 2 : c.near > 0.35 ? 1 : 0
  // Quick to wake, slow to settle.
  if (target > c.alert) {
    c.alert = target
    c.calm = 0
  } else if (target < c.alert) {
    c.calm += dt
    if (c.calm > 1.4) {
      c.alert = target
      c.calm = 0
    }
  } else c.calm = 0
  gaze(c, pointer, heat > 0.3)

  c.shut = Math.max(0, c.shut - dt)
  c.blink -= dt
  if (c.blink <= 0) {
    c.blink = 3 + Math.random() * 3
    if (c.alert > 0) c.shut = 0.15
  }
  c.twitch = impulse ? 0.3 : Math.max(0, c.twitch - dt)
  c.nextFlick -= dt
  if (c.nextFlick <= 0) {
    c.nextFlick = 4 + Math.random() * 5
    c.flick = 0.6
  }
  c.flick = Math.max(0, c.flick - dt)
  c.puff += ((heat > 0.3 ? 1 : 0) - c.puff) * Math.min(1, dt * 4)
}

const ellipse = (x: number, y: number, rx: number, ry: number) =>
  (Math.hypot(x / rx, y / ry) - 1) * Math.min(rx, ry)

function rect(x: number, y: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = Math.max(x0 - x, x - x1)
  const dy = Math.max(y0 - y, y - y1)
  return dx > 0 || dy > 0 ? Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) : Math.max(dx, dy)
}

export function paintCat(
  d: Uint8ClampedArray,
  g: Grid,
  c: Cat,
  ink: CatInk,
  clock: number,
  ground: (x: number) => number,
) {
  const { x: ox, y: oy } = c
  const keep = (x: number, y: number) => y <= ground(x) + 2
  // Everything below is in the cat's own frame: x to its tail, y down, the chest on the ground.
  const local = (f: (x: number, y: number) => number) => (x: number, y: number) => f(x - ox, y - oy)
  const box = (x0: number, y0: number, x1: number, y1: number) =>
    [ox + x0, oy + y0, ox + x1, oy + y1] as const
  const lumpy = (x: number, y: number, seed: number, k: number) =>
    (noise(x * 0.07, y * 0.07, seed) - 0.5) * k

  // Breathing lifts the back by a pixel, in stop-motion.
  const breath = Math.sin(clock * 1.3) > 0.2 ? g.px * 0.8 : 0
  const body = (x: number, y: number) => {
    let f = ellipse(x, y + 30, 62, 30 + breath)
    f = smin(f, Math.hypot(x - 36, y + 24) - 30 - breath * 0.5, 14)
    // The base runs down into the slope, however it falls away under the haunch.
    f = smin(f, rect(x, y, -54, -30, 60, 40), 10)
    return f + lumpy(x, y, 71, 5)
  }
  paintBlob(d, g, local(body), box(-72, -70, 72, 44), ink.stone, keep)
  // A crack down the flank, from when it was rubble.
  const crack: Pt[] = [
    [22, -44],
    [27, -35],
    [22, -27],
    [28, -18],
  ]
  for (let k = 1; k < crack.length; k++)
    line(
      d,
      g,
      col(g, ox + crack[k - 1][0]),
      row(g, oy + crack[k - 1][1]),
      col(g, ox + crack[k][0]),
      row(g, oy + crack[k][1]),
      ink.stone.ink,
    )

  // The head tilts when it is curious; the ears prick when the portal opens, and the near one
  // flattens for a beat at each impulse of the clock.
  const tilt = c.alert === 1 ? 0.14 : 0
  const perk = c.puff * 5
  const flat = c.twitch > 0 ? 1 : 0
  const earA: [Pt, Pt, Pt] = [
    [-86, -62],
    [-80, -88 - perk],
    [-66, -70],
  ]
  const earB: [Pt, Pt, Pt] = [
    [-58, -70],
    [-44 + flat * 8, -88 - perk + flat * 7],
    [-38, -60],
  ]
  const turn = (x: number, y: number): Pt => {
    const dx = x - HEAD[0]
    const dy = y - HEAD[1]
    return [
      HEAD[0] + dx * Math.cos(tilt) + dy * Math.sin(tilt),
      HEAD[1] - dx * Math.sin(tilt) + dy * Math.cos(tilt),
    ]
  }
  const head = (x0: number, y0: number) => {
    const [x, y] = turn(x0, y0)
    let f = Math.hypot(x - HEAD[0], y - HEAD[1]) - 24
    f = smin(f, Math.hypot(x + 75, y + 40) - 13, 8)
    f = smin(f, Math.hypot(x + 50, y + 38) - 13, 8)
    f = smin(f, triangle(x, y, ...earA), 5)
    f = smin(f, triangle(x, y, ...earB), 5)
    // The bib under the chin.
    f = smin(f, Math.hypot(x + 48, y + 20) - 16, 10)
    return f + lumpy(x, y, 73, 3)
  }
  const face = paintBlob(d, g, local(head), box(-100, -104, -24, 0), ink.stone, keep)
  const hi = col(g, ox + HEAD[0])
  const hj = row(g, oy + HEAD[1])
  const pix = (x: number, y: number): Pt => {
    // Local point on the tilted head to its pixel.
    const dx = x - HEAD[0]
    const dy = y - HEAD[1]
    return [
      col(g, ox + HEAD[0] + dx * Math.cos(tilt) - dy * Math.sin(tilt)),
      row(g, oy + HEAD[1] + dx * Math.sin(tilt) + dy * Math.cos(tilt)),
    ]
  }
  if (face) {
    for (const ear of [earA, earB]) {
      const [ei, ej] = pix(
        (ear[0][0] + ear[1][0] + ear[2][0]) / 3,
        (ear[0][1] + ear[1][1] + ear[2][1]) / 3 + 2,
      )
      dot(d, g, ei, ej, ink.ear)
      dot(d, g, ei, ej - 1, ink.ear)
    }
    const k = ink.stone.ink
    const ej = hj - 1
    const shut = c.shut > 0
    const eye = (e: number, open: number) => {
      if (open === 0 || shut) {
        dot(d, g, e - 1, ej, k)
        dot(d, g, e, ej + 1, k)
        dot(d, g, e + 1, ej, k)
        return
      }
      for (let di = -1; di <= 1; di++) {
        dot(d, g, e + di, ej, ink.eye)
        if (open === 2) dot(d, g, e + di, ej - 1, ink.eye)
      }
      const p = e + c.look
      dot(d, g, p, ej, k)
      if (open === 2) dot(d, g, p, ej - 1, k)
      // Round pupils when the portal is open.
      if (open === 2 && c.puff > 0.5) {
        const q = p + (c.look > 0 ? -1 : 1)
        dot(d, g, q, ej, k)
        dot(d, g, q, ej - 1, k)
      }
    }
    eye(hi - 4, c.alert)
    eye(hi + 1, c.alert === 2 ? 2 : 0)
    dot(d, g, hi - 2, hj + 2, ink.nose)
    for (const [di, dj] of [
      [-4, 3],
      [-3, 4],
      [-2, 3],
      [-1, 4],
      [0, 3],
    ])
      dot(d, g, hi + di, hj + dj, k)
    // Whiskers past the cheek.
    line(d, g, hi - 7, hj + 2, hi - 11, hj + 1, k)
    line(d, g, hi - 7, hj + 3, hi - 11, hj + 5, k)
  }

  // The tail wraps the front, its tip lifting when it flicks and fattening when it puffs.
  const lift = c.flick > 0 ? Math.sin((c.flick / 0.6) * Math.PI) * 12 : 0
  const w = 6 + 2 * c.puff
  const tail: Pt[] = [
    [60, -14],
    [50, -4],
    [20, -2],
    [-12, -4],
    [-28, -10 - lift],
  ]
  const tailField = (x: number, y: number) => {
    let f = 1e9
    for (let k = 1; k < tail.length; k++) {
      const r = k === tail.length - 1 ? w + 4 * c.puff : w
      f = smin(f, segment(x, y, tail[k - 1][0], tail[k - 1][1], tail[k][0], tail[k][1]) - r, 6)
    }
    return f + lumpy(x, y, 79, 3)
  }
  paintBlob(d, g, local(tailField), box(-48, -40, 74, 10), ink.stone, keep)
  const paws = (x: number, y: number) =>
    Math.min(ellipse(x + 64, y + 3, 11, 6), ellipse(x + 42, y + 1, 11, 6)) + lumpy(x, y, 83, 2)
  paintBlob(d, g, local(paws), box(-80, -12, -28, 10), ink.stone, keep)

  // Moss on its back and behind its head, flowering.
  const moss = (x: number, y: number) => {
    let f = 1e9
    for (const [bx, by, r] of [
      [8, -60, 12],
      [24, -58, 9],
      [-6, -58, 8],
      [38, -50, 6],
      [-36, -64, 6],
    ])
      f = smin(f, Math.hypot(x - bx, y - by - breath) - r, 7)
    return f + lumpy(x, y, 89, 5)
  }
  const tuft = paintBlob(d, g, local(moss), box(-46, -78, 48, -40), ink.moss, keep)
  if (tuft)
    interior(tuft, (i, j) => {
      const h = hash(i, j, 89)
      if (h < 0.08) put(d, (j * g.bw + i) * 4, ink.flowers[h < 0.04 ? 0 : 1])
    })
}
