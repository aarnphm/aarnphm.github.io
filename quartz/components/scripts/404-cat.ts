// The rubble on the right bank was a cat all along: a loaf of the ruin's stone facing the gate, a
// crack down its flank and moss on its back. It sleeps. A pointer nearby makes it peek with one
// eye; the open portal wakes it with round pupils and a puffed tail. Awake, it blinks, and it
// twitches an ear on every impulse of the clock. Now and then the tip of its tail flicks.
// Every few minutes it gets up and walks the crest, up toward the gate or down toward the water,
// stands there a while with its tail up, and comes home to lie down again facing the gate. A fish
// thrown up the bank from the boat (404-angler.ts) it eats where it lies, eyes shut, and leaves the
// bones.
// It is a British Blue: round in the face and full in the cheek, small ears set wide, a short neck,
// short thick legs and a thick tail. On a slope its back takes most of the pitch and its legs the
// rest, each leg reaching from the hip to where its foot comes down on the ground.
// Each part is its own blob, painted back to front, so the far legs, the head, the tail and the
// paws are outlined where they lie against the body.

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
  stamp,
  triangle,
} from './404-pixel'

type Doing = 'lie' | 'rise' | 'turn' | 'walk' | 'stand' | 'settle'

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
  // -1 facing the gate, as it lies at home.
  face: number
  doing: Doing
  t: number
  // Where it is walking to, how long it stands there, and seconds until it next gets up.
  to: number
  until: number
  roam: number
  // 0 lying, 1 on its feet; the walk's progress in strides; the head down to eat.
  stand: number
  gait: number
  dip: number
  // Where its frame sits on the ground, and how far it pitches with the slope under its feet.
  oy: number
  pitch: number
  // A thrown fish on its way, the one it is eating (1 whole, 0 gone), and the bones it leaves.
  coming: boolean
  meal: { x: number; y: number; left: number } | null
  bones: number
}

export type CatInk = {
  stone: Tones
  moss: Tones
  eye: RGB
  nose: RGB
  ear: RGB
  flowers: RGB[]
  fish: RGB
}

// Where the chest meets the ground, and the head relative to it: lying, standing, and down at a
// fish in front of its paws.
const HEAD: Pt = [-62, -50]
const UP_HEAD: Pt = [-62, -74]
const DOWN_HEAD: Pt = [-80, -26]
// Standing, its body rises and slims this much. Each leg is two bones of this length; a foot swings
// this far either side of under its hip and lifts this high.
const RISE = 20
const THIN = 2
const BONE = 22
const SWING = 13
const LIFT = 6
// The back takes this share of the slope under it and the legs the rest. The head turns back
// against the back's pitch by this share, about the shoulders.
const LEAN = 0.85
const NOD = 0.5
// The shelf on the flank it lies on, the spots it walks to up and down the crest, its pace in units
// a second, and the ground covered in a full cycle of its legs.
const HOME = 1250
const UPHILL = 1140
const DOWNHILL = 1385
const PACE = 12
const STRIDE = 52
// Where a thrown fish lands, in front of its paws.
const MEAL = 104

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
    face: -1,
    doing: 'lie',
    t: 0,
    to: x,
    until: 0,
    roam: 60 + Math.random() * 240,
    stand: 0,
    gait: 0,
    dip: 0,
    oy: ground + 2,
    pitch: 0,
    coming: false,
    meal: null,
    bones: 0,
  }
}

// A local point on the cat, x toward its tail and y down, to the scene.
function world(c: Cat, x: number, y: number): Pt {
  const s = -c.face
  const u = x * Math.cos(c.pitch) - y * Math.sin(c.pitch)
  const v = x * Math.sin(c.pitch) + y * Math.cos(c.pitch)
  return [c.x + u * s, c.oy + v]
}

// Where a foot comes down: its point along the body, on the ground under it, lifted mid-swing.
function footing(c: Cat, x: number, lift: number, ground: (x: number) => number): Pt {
  const wx = c.x - c.face * x * Math.cos(c.pitch)
  return [wx, ground(wx) - 3 - lift]
}

// The body's centre and half-height in its own frame, from lying to standing.
const trunk = (stand: number) => ({ cy: -30 - RISE * stand, ry: 30 - THIN * stand })

const spin = ([x, y]: Pt, [ox, oy]: Pt, a: number): Pt => [
  ox + (x - ox) * Math.cos(a) - (y - oy) * Math.sin(a),
  oy + (x - ox) * Math.sin(a) + (y - oy) * Math.cos(a),
]

const headAt = (c: Cat): Pt => [
  HEAD[0] + (UP_HEAD[0] - HEAD[0]) * c.stand + (DOWN_HEAD[0] - HEAD[0]) * c.dip,
  HEAD[1] + (UP_HEAD[1] - HEAD[1]) * c.stand + (DOWN_HEAD[1] - HEAD[1]) * c.dip,
]

// The head's shapes are drawn about HEAD and carried to headAt, then turned about the shoulders
// against the back's pitch, so on a slope the head stays nearer level than the back.
function carry(c: Cat) {
  const [hx, hy] = headAt(c)
  const shoulder: Pt = [-40, trunk(c.stand).cy - 8]
  return { mx: hx - HEAD[0], my: hy - HEAD[1], q: -NOD * c.pitch, shoulder }
}

export const catHead = (c: Cat): Pt => {
  const { mx, my, q, shoulder } = carry(c)
  return world(c, ...spin([HEAD[0] + mx, HEAD[1] + my], shoulder, q))
}

// Where a fish thrown to it should land: in front of its paws while it lies at home with nothing
// to eat, or nowhere.
export function mealFor(c: Cat, ground: (x: number) => number): Pt | null {
  if (c.doing !== 'lie' || c.stand > 0 || c.meal || c.coming || c.bones > 0 || c.puff > 0.2)
    return null
  const x = c.x - MEAL
  return [x, ground(x) + 2]
}

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
  ground: (x: number) => number,
) {
  const hot = heat > 0.3
  c.near = near(c, pointer) ? c.near + dt : 0
  // Up and about, or with a fish coming, it is wide awake.
  const busy = c.stand > 0.3 || c.coming || !!c.meal
  const target = hot || busy ? 2 : c.near > 0.35 ? 1 : 0
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
  gaze(c, pointer, hot)

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
  c.puff += ((hot ? 1 : 0) - c.puff) * Math.min(1, dt * 4)

  // Head down to the fish, eaten in five seconds, and the bones left a while.
  c.bones = Math.max(0, c.bones - dt)
  if (c.meal) {
    c.dip = Math.min(1, c.dip + dt / 0.6)
    if (c.dip >= 1) c.meal.left -= dt / 5
    if (c.meal.left <= 0) {
      c.meal = null
      c.bones = 40
    }
  } else c.dip = Math.max(0, c.dip - dt / 0.6)

  stepWalk(c, dt, hot)
  // Once it is up, its frame stands on the ground under its feet, pitched with part of the slope.
  // Where a downhill foot cannot reach the ground, the body comes down until it can.
  const s = -c.face
  const front = ground(c.x - 40 * s)
  const hind = ground(c.x + 40 * s)
  c.y = ground(c.x) + 2
  c.oy = c.y + ((front + hind) / 2 + 2 - c.y) * c.stand
  c.pitch = LEAN * Math.atan2(hind - front, 80) * c.stand
  if (c.stand <= 0) return
  const { cy, ry } = trunk(1)
  let reach = 0
  for (const l of LEGS)
    for (const dx of [-SWING, SWING]) {
      const [hx, hy] = world(c, l.x, cy + 0.25 * ry)
      const [fx, fy] = footing(c, l.x + dx, 0, ground)
      reach = Math.max(reach, Math.hypot(fx - hx, fy - hy))
    }
  c.oy += Math.max(0, reach - 1.96 * BONE) * c.stand
}

function stepWalk(c: Cat, dt: number, hot: boolean) {
  c.t += dt
  const go = (doing: Doing) => {
    c.doing = doing
    c.t = 0
  }
  // Which way it should face for what comes next: toward where it is going, or the gate at home.
  const want = c.x === c.to ? -1 : Math.sign(c.to - c.x)
  switch (c.doing) {
    case 'lie':
      c.stand = Math.max(0, c.stand - dt / 1.5)
      if (hot || c.coming || c.meal || (c.roam -= dt) > 0) break
      c.to = Math.random() < 0.5 ? UPHILL : DOWNHILL
      go('rise')
      break
    case 'rise':
      c.stand = Math.min(1, c.stand + dt / 1.2)
      if (c.stand >= 1) go(want === c.face ? 'walk' : 'turn')
      break
    case 'turn':
      if (c.t < 0.6) break
      c.face = want
      go(c.x === c.to ? 'settle' : 'walk')
      break
    case 'walk': {
      // It stops where it is while the portal is open.
      if (hot) break
      const gap = c.to - c.x
      const step = Math.min(Math.abs(gap), PACE * dt)
      c.x += Math.sign(gap) * step
      c.gait += step / STRIDE
      if (Math.abs(c.to - c.x) > 0.01) break
      c.x = c.to
      if (c.to === HOME) go(c.face === -1 ? 'settle' : 'turn')
      else {
        c.until = 15 + Math.random() * 25
        go('stand')
      }
      break
    }
    case 'stand':
      if (c.t < c.until || hot) break
      c.to = HOME
      go(Math.sign(HOME - c.x) === c.face ? 'walk' : 'turn')
      break
    case 'settle':
      c.stand = Math.max(0, c.stand - dt / 1.5)
      if (c.stand > 0) break
      c.roam = 150 + Math.random() * 210
      go('lie')
      break
  }
}

const ellipse = (x: number, y: number, rx: number, ry: number) =>
  (Math.hypot(x / rx, y / ry) - 1) * Math.min(rx, ry)

function rect(x: number, y: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = Math.max(x0 - x, x - x1)
  const dy = Math.max(y0 - y, y - y1)
  return dx > 0 || dy > 0 ? Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) : Math.max(dx, dy)
}

const lerp = (a: Pt, b: Pt, t: number): Pt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]

// The legs in a walk: near hind, near fore, far hind, far fore, a quarter of a cycle apart. Each
// foot swings forward clear of the ground and stays planted while the body passes over it.
const LEGS = [
  { x: 34, off: 0, far: false },
  { x: -30, off: 0.25, far: false },
  { x: 26, off: 0.5, far: true },
  { x: -22, off: 0.75, far: true },
]
// The fish, eaten from the head down to the bones.
const FISH = [['F..FF.', '.FFFFF', 'F..FF.'], ['F..F..', '.FFF..', 'F..F..'], ['F.....']]
const BONES = ['F.F.F.', 'FFFFFF', 'F.F.F.']

export function paintCat(
  d: Uint8ClampedArray,
  g: Grid,
  c: Cat,
  ink: CatInk,
  clock: number,
  ground: (x: number) => number,
) {
  const s = -c.face
  const keep = (x: number, y: number) => y <= ground(x) + 2
  const cos = Math.cos(c.pitch)
  const sin = Math.sin(c.pitch)
  // Everything below is in the cat's own frame: x to its tail, y down, the chest on the ground.
  const inward = (x: number, y: number): Pt => {
    const u = (x - c.x) * s
    const v = y - c.oy
    return [u * cos + v * sin, -u * sin + v * cos]
  }
  const local = (f: (x: number, y: number) => number) => (x: number, y: number) =>
    f(...inward(x, y))
  const box = (x0: number, y0: number, x1: number, y1: number) => {
    const corners = [world(c, x0, y0), world(c, x1, y0), world(c, x0, y1), world(c, x1, y1)]
    const xs = corners.map(p => p[0])
    const ys = corners.map(p => p[1])
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] as const
  }
  const pixel = (x: number, y: number): Pt => {
    const [wx, wy] = world(c, x, y)
    return [col(g, wx), row(g, wy)]
  }
  const lumpy = (x: number, y: number, seed: number, k: number) =>
    (noise(x * 0.07, y * 0.07, seed) - 0.5) * k

  // The fish in front of its paws, under the head that comes down to it, then its bones.
  const dish =
    c.meal ?? (c.bones > 0 ? { x: c.x - MEAL, y: ground(c.x - MEAL) + 2, left: 0 } : null)
  if (dish) {
    const rows = dish.left > 0 ? FISH[dish.left > 0.6 ? 0 : dish.left > 0.25 ? 1 : 2] : BONES
    const tone = dish.left > 0 ? ink.fish : ink.stone.light
    stamp(d, g, rows, col(g, dish.x) - 3, row(g, dish.y) - 3, s < 0, () => tone)
  }

  const stand = c.stand
  // Breathing lifts the back by a pixel, in stop-motion.
  const breath = Math.sin(clock * 1.3) > 0.2 ? g.px * 0.8 : 0
  const { cy, ry } = trunk(stand)
  // Each leg reaches from its hip to where its foot comes down, bending at the wrist toward the
  // head in front and at the hock toward the tail behind. The legs swing in eight steps to a
  // stride.
  const gait = Math.floor(c.gait * 8) / 8
  const legs =
    stand < 0.5
      ? []
      : LEGS.map(l => {
          const a = 2 * Math.PI * (gait + l.off)
          const hip: Pt = [l.x, cy + 0.25 * ry]
          const lift = LIFT * Math.max(0, Math.cos(a))
          const [fx, fy] = inward(...footing(c, l.x - SWING * Math.sin(a), lift, ground))
          const len = Math.hypot(fx - hip[0], fy - hip[1])
          const ux = (fx - hip[0]) / len
          const uy = (fy - hip[1]) / len
          const span = Math.min(len, 2 * BONE - 0.01)
          const bow = Math.sqrt(BONE * BONE - (span / 2) ** 2) * (l.x < 0 ? -1 : 1)
          const knee: Pt = [
            hip[0] + (ux * span) / 2 + uy * bow,
            hip[1] + (uy * span) / 2 - ux * bow,
          ]
          const foot: Pt = [hip[0] + ux * span, hip[1] + uy * span]
          return { hip, knee, foot, far: l.far }
        })
  type Leg = (typeof legs)[number]
  const limb = (x: number, y: number, l: Leg) => {
    const f = smin(
      segment(x, y, ...l.hip, ...l.knee) - 10,
      segment(x, y, ...l.knee, ...l.foot) - 8,
      4,
    )
    return smin(f, ellipse(x - l.foot[0] + 3, y - l.foot[1], 10, 6), 4)
  }
  // The far legs behind the body, a tone darker.
  const far = legs.filter(l => l.far)
  if (far.length)
    paintBlob(
      d,
      g,
      local((x, y) => Math.min(...far.map(l => limb(x, y, l))) + lumpy(x, y, 67, 3)),
      box(-72, -60, 72, 64),
      { ...ink.stone, light: ink.stone.plate, plate: ink.stone.shade },
      keep,
    )
  const body = (x: number, y: number) => {
    // Standing, it is short in the body, deep in the chest and round in the haunch.
    let f = ellipse(x, y - cy, 62 - 12 * stand, ry + breath)
    f = smin(f, Math.hypot(x + 34 - 2 * stand, y - cy - 4) - 26, 14)
    f = smin(f, Math.hypot(x - 36 + 6 * stand, y - cy - 6) - 30 + 4 * stand - breath * 0.5, 14)
    // Lying, the base runs down into the slope, however it falls away under the haunch.
    if (stand < 0.5) f = smin(f, rect(x, y, -54, -30, 60, 40), 10)
    else for (const l of legs) if (!l.far) f = smin(f, limb(x, y, l), 6)
    return f + lumpy(x, y, 71, 5)
  }
  paintBlob(d, g, local(body), box(-72, -96, 72, 64), ink.stone, keep)
  // A crack down the flank, from when it was rubble.
  const crack: Pt[] = [
    [22, -44],
    [27, -35],
    [22, -27],
    [28, -18],
  ]
  for (let k = 1; k < crack.length; k++) {
    const [i0, j0] = pixel(crack[k - 1][0], crack[k - 1][1] + cy + 30)
    const [i1, j1] = pixel(crack[k][0], crack[k][1] + cy + 30)
    line(d, g, i0, j0, i1, j1, ink.stone.ink)
  }

  // The head tilts when it is curious; the ears prick when the portal opens, and the near one
  // flattens for a beat at each impulse of the clock.
  const { mx, my: carried, q, shoulder } = carry(c)
  const my = carried + (stand > 0.5 && Math.floor(c.gait * 8) % 2 ? g.px : 0)
  const tilt = c.alert === 1 && stand === 0 ? 0.14 : 0
  const perk = c.puff * 4
  const flat = c.twitch > 0 ? 1 : 0
  // Upright ears on the corners of the crown, set wide and rounded at the tips.
  const earA: [Pt, Pt, Pt] = [
    [-90, -60],
    [-85, -83 - perk],
    [-68, -69],
  ]
  const earB: [Pt, Pt, Pt] = [
    [-56, -69],
    [-39 + flat * 6, -83 - perk + flat * 6],
    [-34, -60],
  ]
  // A point drawn about HEAD to where the head is now, and back.
  const out = (x: number, y: number): Pt => {
    const [tx, ty] = spin([x, y], HEAD, tilt)
    return spin([tx + mx, ty + my], shoulder, q)
  }
  const rest = (x: number, y: number): Pt => {
    const [ux, uy] = spin([x, y], shoulder, -q)
    return spin([ux - mx, uy - my], HEAD, -tilt)
  }
  const head = (x0: number, y0: number) => {
    const [x, y] = rest(x0, y0)
    // A round face turned to us, widest at the full cheeks, with no muzzle to speak of.
    let f = ellipse(x - HEAD[0], y - HEAD[1] - 1, 31, 25)
    f = smin(f, ellipse(x - HEAD[0], y - HEAD[1] - 9, 36, 16), 8)
    f = smin(f, triangle(x, y, ...earA) - 3, 5)
    f = smin(f, triangle(x, y, ...earB) - 3, 5)
    return f + lumpy(x, y, 73, 3)
  }
  const [ox, oy] = out(...HEAD)
  const face = paintBlob(
    d,
    g,
    local(head),
    box(ox - 50, oy - 50, ox + 50, oy + 40),
    ink.stone,
    keep,
  )
  // A point drawn about HEAD to its pixel.
  const pix = (x: number, y: number): Pt => pixel(...out(x, y))
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
    const [hi, hj] = pix(HEAD[0], HEAD[1])
    const ej = hj - 1
    const shut = c.shut > 0
    // Eating, the eyes are shut tight and turned up at the ends.
    const eating = c.dip > 0.5
    const awake = stand > 0.3 ? 2 : c.alert
    const eye = (e: number, open: number) => {
      if (eating) {
        dot(d, g, e - 1, ej + 1, k)
        dot(d, g, e, ej, k)
        dot(d, g, e + 1, ej + 1, k)
        return
      }
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
    // Big round eyes either side of a short nose, and the mouth under it working while it eats.
    eye(hi - 3 * s, awake)
    eye(hi + 3 * s, awake === 2 ? 2 : 0)
    dot(d, g, hi, hj + 2, ink.nose)
    const chew = eating && Math.floor(clock * 4) % 2 ? 1 : 0
    for (const [di, dj] of [
      [-2, 3],
      [-1, 4],
      [0, 3 + chew],
      [1, 4],
      [2, 3],
    ])
      dot(d, g, hi + di, hj + dj, k)
    // Whiskers out past both cheeks.
    for (const side of [-1, 1]) {
      line(d, g, hi + 6 * side, hj + 2, hi + 10 * side, hj + 1, k)
      line(d, g, hi + 6 * side, hj + 3, hi + 10 * side, hj + 5, k)
    }
  }

  // The tail: lying, it wraps the front, its tip lifting when it flicks and fattening when it puffs;
  // on its feet, it is carried upright with the tip turned over. It is thick at the root and stays
  // thick to a round tip.
  const lift = c.flick > 0 ? Math.sin((c.flick / 0.6) * Math.PI) * 12 : 0
  const w = 7 + 2 * c.puff
  const down: Pt[] = [
    [60, -14],
    [50, -4],
    [20, -2],
    [-12, -4],
    [-26, -10 - lift],
  ]
  const raised: Pt[] = [
    [48, -60],
    [58, -74],
    [62, -90],
    [61, -106],
    [55 - lift * 0.5, -116 - lift * 0.5],
  ]
  const tail = down.map((p, k) => lerp(p, raised[k], stand))
  const tailField = (x: number, y: number) => {
    let f = 1e9
    for (let k = 1; k < tail.length; k++) {
      const r = w + (k === 1 ? 2 : 0) + (k === tail.length - 1 ? 1 + 4 * c.puff : 0)
      f = smin(f, segment(x, y, tail[k - 1][0], tail[k - 1][1], tail[k][0], tail[k][1]) - r, 6)
    }
    return f + lumpy(x, y, 79, 3)
  }
  paintBlob(d, g, local(tailField), box(-48, -132, 98, 10), ink.stone, keep)
  if (stand < 0.5) {
    const paws = (x: number, y: number) =>
      Math.min(ellipse(x + 64, y + 3, 11, 6), ellipse(x + 42, y + 1, 11, 6)) + lumpy(x, y, 83, 2)
    paintBlob(d, g, local(paws), box(-80, -12, -28, 10), ink.stone, keep)
  }

  // Moss on its back and behind its head, flowering.
  const back = cy + 30
  const moss = (x: number, y: number) => {
    let f = 1e9
    for (const [bx, by, r] of [
      [8, -60, 12],
      [24, -58, 9],
      [-6, -58, 8],
      [38, -50, 6],
      [-36, -64, 6],
    ])
      f = smin(f, Math.hypot(x - bx, y - by - breath - back) - r, 7)
    return f + lumpy(x, y, 89, 5)
  }
  const tuft = paintBlob(d, g, local(moss), box(-46, -78 + back, 48, -40 + back), ink.moss, keep)
  if (tuft)
    interior(tuft, (i, j) => {
      const h = hash(i, j, 89)
      if (h < 0.08) put(d, (j * g.bw + i) * 4, ink.flowers[h < 0.04 ? 0 : 1])
    })
}
