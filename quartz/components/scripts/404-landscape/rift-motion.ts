import { GATE } from '../404-gate'
import { fbm, hex, mulberry, type Grid, type Pt, type RGB } from '../404-pixel'

const STAR_COLORS = ['#fffcf0', '#fffcf0', '#f1d67e', '#fdb2a2', '#cdd597', '#92bfdb'].map(hex)

export const LEAK = 58
const RIM = GATE.r + LEAK + 4

const GM = 4.5e6

type Mote = {
  r: number
  a: number
  tone: number
  thr: number
  len: number
  wide: boolean
  star: RGB | null
  mag: number
  freq: number
  phase: number
}
type Comet = {
  x: number
  y: number
  vx: number
  vy: number
  age: number
  acc: number
  trail: number[]
}
export type Opening = {
  grid: Grid
  mask: Uint8Array
  i0: number
  i1: number
  j0: number
  j1: number
}
export type Rift = {
  opening: Opening | null
  pole: Pt
  motes: Mote[]
  rate: number
  target: number
  inflow: number
  inflowTarget: number
  pattern: number
  heat: number
  flare: number
  comets: Comet[]
  nextComet: number
}

// Differential rotation: the core turns about twice as fast as the reference radius, the rim half.
export const orbit = (r: number) => 180 / (r + 90)

function nebula(x: number, y: number) {
  const q = fbm(x, y, 211, 3)
  const r = fbm(x + 4 * q, y + 4 * q, 223, 3)
  return fbm(x + 3.5 * r, y + 3.5 * r, 227, 3)
}

export function makeRift(pole: Pt): Rift {
  const rand = mulberry(404)
  const motes = Array.from({ length: 1600 }, () => {
    const r = RIM * Math.sqrt(rand())
    const a = rand() * Math.PI * 2
    const star = rand() < 0.05 ? STAR_COLORS[Math.floor(rand() * STAR_COLORS.length)] : null
    return {
      r,
      a,
      tone: nebula(Math.cos(a) * r * 0.006, Math.sin(a) * r * 0.006),
      thr: rand(),
      len: star ? 1 : 3 + Math.floor(rand() * 4),
      wide: !star && rand() < 0.45,
      star,
      mag: rand() ** 2,
      freq: 0.8 + rand() * 2.4,
      phase: rand() * Math.PI * 2,
    }
  })
  return {
    opening: null,
    pole,
    motes,
    rate: 1.1,
    target: 1.1,
    inflow: 0,
    inflowTarget: 0,
    pattern: 0,
    heat: 0,
    flare: 0,
    comets: [],
    nextComet: 2.5,
  }
}

export function stepRift(rift: Rift, dt: number) {
  rift.rate += (rift.target - rift.rate) * Math.min(1, dt * 2.2)
  rift.inflow += (rift.inflowTarget - rift.inflow) * Math.min(1, dt * 1.6)
  rift.heat = Math.min(1, Math.max(0, (rift.rate - 1.1) / 24))
  rift.flare *= Math.exp(-dt * 2.5)
  const spin = (rift.rate * Math.PI) / 180
  rift.pattern -= spin * 0.6 * dt
  // Motes keep their radius so the disc never thins out; the inflow shows only as pitch.
  for (const m of rift.motes) m.a += orbit(m.r) * spin * dt

  rift.nextComet -= dt
  if (rift.nextComet <= 0) {
    // Launched from beyond the ring, mostly sideways and a little inward, so some orbit and some
    // escape.
    const a = Math.random() * Math.PI * 2
    const v = 130 + Math.random() * 170
    rift.comets.push({
      x: rift.pole[0] + Math.cos(a) * (GATE.r + 40),
      y: rift.pole[1] + Math.sin(a) * (GATE.r + 40),
      vx: -Math.sin(a) * v - Math.cos(a) * 40,
      vy: Math.cos(a) * v - Math.sin(a) * 40,
      age: 0,
      acc: 0,
      trail: [],
    })
    rift.nextComet = rift.heat > 0.5 ? 0.5 + Math.random() * 0.8 : 3 + Math.random() * 5
  }
  // Inverse-square pull toward the pole bends every path into a conic; the inflow adds drag, so
  // on hover the comets spiral in and the pole flares as it swallows them.
  const drag = Math.exp(-rift.inflow * 1.4 * dt)
  let cw = 0
  for (let i = 0; i < rift.comets.length; i++) {
    const c = rift.comets[i]
    const dx = c.x - rift.pole[0]
    const dy = c.y - rift.pole[1]
    const r2 = dx * dx + dy * dy + 300
    const pull = GM / (r2 * Math.sqrt(r2))
    c.vx = (c.vx - dx * pull * dt) * drag
    c.vy = (c.vy - dy * pull * dt) * drag
    c.x += c.vx * dt
    c.y += c.vy * dt
    c.age += dt
    c.acc += dt
    if (c.acc >= 1 / 60) {
      c.acc = 0
      c.trail.push(c.x, c.y)
      if (c.trail.length > 44) c.trail.splice(0, 2)
    }
    if (r2 < 450) {
      c.age = 99
      rift.flare = 1
    } else if (r2 > 520 * 520) c.age = 99
    if (c.age < 3.2) rift.comets[cw++] = c
  }
  rift.comets.length = cw
}
