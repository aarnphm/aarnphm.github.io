import type { Pt } from '../404-pixel'
import type { Park, Rider, Walk } from './park-state'
import { along, PATHS, seek, surface, TRACKS, type Path, type Track } from '../404-downhill'
import { HUT } from './rescue'

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

export const onCourse = (r: Rider) =>
  r.state === 'ride' || r.state === 'air' || r.state === 'down' || r.state === 'up'
export const mainQ = (r: Rider) => {
  const t = TRACKS[r.line]
  const k = Math.min(t.n - 2, r.k)
  return t.main[k] + ((r.q - t.q[k]) * (t.main[k + 1] - t.main[k])) / (t.q[k + 1] - t.q[k])
}

const isTrack = (on: Path | Track): on is Track => 'n' in on

// On a line at `q`: the point, and which way the line runs there.
// Across a line at a sample, square to the way down it on the screen.
export function across(t: Track, k: number): Pt {
  const j = Math.min(t.n - 2, k)
  const dx = t.x[j + 1] - t.x[j]
  const dy = t.y[j + 1] - t.y[j]
  const l = Math.hypot(dx, dy) || 1
  return [-dy / l, dx / l]
}

export function onTrack(t: Track, q: number) {
  const k = Math.min(t.n - 2, seek(t, q, 0))
  const u = clamp((q - t.q[k]) / (t.q[k + 1] - t.q[k]), 0, 1)
  return {
    x: t.x[k] + (t.x[k + 1] - t.x[k]) * u,
    y: -surface(t, q, k),
    face: Math.sign(t.x[Math.min(t.n - 1, k + 3)] - t.x[Math.max(0, k - 3)]) || 1,
  }
}

export function walkAt(w: Walk) {
  const l = w.legs[w.n]
  const dir = Math.sign(l.to - l.from) || 1
  const s = l.from + dir * w.u
  if (isTrack(l.on)) {
    const at = onTrack(l.on, s)
    return { x: at.x, y: at.y, face: at.face * dir }
  }
  const at = along(l.on, s)
  return { x: at.x, y: at.y, face: at.dir * dir }
}

// Where along the main line someone on foot on a line is, or null when they are off it.
export function footQ(w: Walk | null) {
  const l = w?.legs[w.n]
  if (!w || !l || !isTrack(l.on)) return null
  const t = l.on
  const q = l.from + (Math.sign(l.to - l.from) || 1) * w.u
  const k = Math.min(t.n - 2, seek(t, q, 0))
  return t.main[k] + ((q - t.q[k]) * (t.main[k + 1] - t.main[k])) / (t.q[k + 1] - t.q[k])
}

// The line stays shut while anyone hurt is still on it.
export const held = (p: Park) =>
  p.riders.some(
    o =>
      o.state === 'hurt' ||
      ((o.state === 'limp' || o.state === 'carried') && footQ(o.walk) !== null),
  )

// Where on the screen a rider is and which way they face, and, in the air, the ground under them.
export function place(r: Rider): { x: number; y: number; face: number; ground: number } {
  const { base, lift, link, runout } = PATHS
  const on = (s: number, p: typeof base) => {
    const at = along(p, s)
    return { x: at.x, y: at.y, face: at.dir, ground: at.y }
  }
  switch (r.state) {
    case 'arrive':
    case 'liftq':
      return { ...on(r.s, base), face: 1 }
    case 'leave':
      return { ...on(r.s, base), face: -1 }
    case 'lift':
      return on(r.s, lift)
    // Beside the lift's track, out of the way of its bars.
    case 'hike': {
      const at = on(r.s, lift)
      return { ...at, x: at.x - 8 }
    }
    case 'link':
    case 'gate':
      return on(r.s, link)
    case 'runout':
      return on(r.s, runout)
    case 'limp':
    case 'carried': {
      const at = walkAt(r.walk!)
      return { ...at, ground: at.y }
    }
    // On the grass beside the hut's door.
    case 'rest':
      return { x: HUT[0] + 24, y: HUT[1], face: -1, ground: HUT[1] }
    default: {
      const t = TRACKS[r.line]
      const at = onTrack(t, r.q)
      const [nx, ny] = across(t, r.k)
      const s = r.state === 'ride' ? r.swerve : 0
      const y = at.y + ny * s
      return { x: at.x + nx * s, y: r.state === 'air' ? -r.z : y, face: at.face, ground: y }
    }
  }
}
