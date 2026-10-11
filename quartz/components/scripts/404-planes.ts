// Now and then something crosses the sky above the ranges: an airliner at cruise, a few pixels long,
// drawing a contrail that lasts as long as the air up there is damp enough to keep it (a clear sky
// loses it within seconds; a sky filling with cloud keeps it, spreading and drifting with the
// wind), or by day a light plane lower down, quicker across. Both go behind the clouds. At night
// only their lights show: red and green at the wingtips, the red beacon, and the white strobe.

import { type Grid, type RGB, BAYER, col, row, smooth } from './404-pixel'

type Kind = 'jet' | 'prop'
type Puff = { x: number; y: number; age: number; drift: number }
type Plane = { kind: Kind; x: number; y: number; dir: number; v: number; climb: number }

export type Skyway = { planes: Plane[]; trail: Puff[]; next: number; drop: number }

export type SkyEnv = {
  cover: number
  day: boolean
  // No light plane goes up in fog or a storm.
  flyable: boolean
  // How far the clouds have drifted, and the scene's edges.
  drift: number
  left: number
  right: number
}

export type PlaneInk = { body: RGB; glint: RGB; trail: RGB; red: RGB; green: RGB; strobe: RGB }

// Scene units per second, and seconds between one and the next.
const SPEED: Record<Kind, number> = { jet: 26, prop: 44 }
const QUIET = { min: 90, span: 240 }

export function makeSkyway(): Skyway {
  return { planes: [], trail: [], next: 20 + Math.random() * 120, drop: 0 }
}

// A contrail lasts longer the cloudier the sky, which stands in for how damp the air is up there.
const life = (cover: number) => 3 + 80 * smooth(0.2, 0.75, cover)

export function stepSkyway(s: Skyway, dt: number, env: SkyEnv) {
  const keep = life(env.cover)
  for (const p of s.trail) p.age += dt
  s.trail = s.trail.filter(p => p.age < keep)
  for (const p of s.planes) {
    p.x += p.dir * p.v * dt
    p.y += p.climb * p.v * dt
    if (p.kind === 'jet' && (s.drop -= dt) <= 0) {
      s.drop = 0.2
      s.trail.push({ x: p.x, y: p.y, age: 0, drift: env.drift })
    }
  }
  s.planes = s.planes.filter(p => p.x > env.left - 60 && p.x < env.right + 60)
  if (s.planes.length || (s.next -= dt) > 0) return
  s.next = QUIET.min + Math.random() * QUIET.span
  // Above an overcast there is nothing to see.
  if (env.cover > 0.85) return
  const kind: Kind =
    env.day && env.flyable && env.cover < 0.7 && Math.random() < 0.3 ? 'prop' : 'jet'
  const dir = Math.random() < 0.5 ? -1 : 1
  s.planes.push({
    kind,
    x: dir > 0 ? env.left - 50 : env.right + 50,
    y: kind === 'jet' ? 70 + Math.random() * 130 : 250 + Math.random() * 70,
    dir,
    v: SPEED[kind] * (0.9 + Math.random() * 0.2),
    climb: (Math.random() - 0.5) * (kind === 'jet' ? 0.05 : 0.1),
  })
}

// Facing right: the fin, the fuselage and the near wing; the light plane's wing over its cabin.
const SHAPE: Record<Kind, string[]> = { jet: ['K...', 'KKKK', '..K.'], prop: ['K.KKK', 'KKKKK'] }

export function paintPlanes(
  d: Uint8ClampedArray,
  g: Grid,
  s: Skyway,
  ink: PlaneInk,
  clock: number,
  night: boolean,
  drift: number,
  cover: number,
  cloud: (x: number, y: number) => number,
) {
  const plot = (i: number, j: number, c: RGB, alpha = 1) => {
    if (i < 0 || j < 0 || i >= g.bw || j >= g.bh) return
    if (alpha <= BAYER[(j & 3) * 4 + (i & 3)]) return
    if (cloud(g.sx[i], g.sy[j]) > 0.45) return
    const k = (j * g.bw + i) * 4
    d[k] = c[0]
    d[k + 1] = c[1]
    d[k + 2] = c[2]
  }
  // The trail, fading and spreading as it ages, carried along with the cloud. It starts a little
  // way behind the engines.
  if (!night) {
    const keep = life(cover)
    for (let n = 1; n < s.trail.length; n++) {
      const a = s.trail[n - 1]
      const b = s.trail[n]
      if (b.age < 0.4 || Math.abs(a.x - b.x) > 20) continue
      const fade = (1 - a.age / keep) ** 0.7
      const i0 = col(g, a.x + drift - a.drift)
      const i1 = col(g, b.x + drift - b.drift)
      const j0 = row(g, a.y)
      const j1 = row(g, b.y)
      const steps = Math.max(Math.abs(i1 - i0), Math.abs(j1 - j0), 1)
      for (let k = 0; k <= steps; k++) {
        const i = Math.round(i0 + ((i1 - i0) * k) / steps)
        const j = Math.round(j0 + ((j1 - j0) * k) / steps)
        plot(i, j, ink.trail, fade)
        if (a.age > 8) plot(i, j + 1, ink.trail, fade * 0.5)
      }
    }
  }
  for (const p of s.planes) {
    const rows = SHAPE[p.kind]
    const w = rows[0].length
    const i0 = col(g, p.x) - (w >> 1)
    const j0 = row(g, p.y) - 1
    const flip = p.dir < 0
    if (!night) {
      rows.forEach((text, r) => {
        for (let c = 0; c < w; c++)
          if (text[flip ? w - 1 - c : c] !== '.') plot(i0 + c, j0 + r, ink.body)
      })
      // The sun catching it now and then.
      if (Math.floor(clock * 2) % 5 === 0) plot(i0 + (flip ? 0 : w - 1), j0 + 1, ink.glint)
      continue
    }
    // Port red and starboard green at the wingtips, a red beacon pulsing on the fuselage, and the
    // white strobes in a double flash.
    const mid = i0 + (w >> 1)
    plot(mid - 1, j0 + 1, p.dir > 0 ? ink.red : ink.green)
    plot(mid + 1, j0 + 1, p.dir > 0 ? ink.green : ink.red)
    if (clock % 1.1 < 0.25) plot(mid, j0, ink.red)
    const flash = clock % 1.4
    if (flash < 0.08 || (flash > 0.2 && flash < 0.28)) plot(mid, j0 + 2, ink.strobe)
  }
}
