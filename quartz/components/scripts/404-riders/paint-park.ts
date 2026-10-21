import type { Voices } from '../404-crew'
import type { Medic, Park, RiderInk } from './park-state'
import { along, MAIN, PATHS, PHOTO, TRACKS } from '../404-downhill'
import { BARN, paintBarn, paintBeast, paintFarmer, paintTrough } from '../404-herd'
import { BAYER, col, dot, row, type Grid, type Pt, type RGB } from '../404-pixel'
import { paintRider, type Pen } from './paint-rider'
import { held, onTrack, place, walkAt } from './positions'
import { HUT } from './rescue'
import { BARS, LIFT_V } from './traffic'

// The riders, the lift's bars, the gate's beacon and the photographer, over the ranges. Each rider
// is a pixel or three wide: wheels and frame, then shorts, jersey and helmet; a child a pixel
// shorter. Tricks move those pixels about: the back wheel kicked up and out for a whip, both arms
// out for a no-hander, the body stretched out flat behind the bars for a superman.
export function paintPark(
  d: Uint8ClampedArray,
  g: Grid,
  ink: RiderInk,
  p: Park,
  dark: boolean,
  heads: Voices['heads'],
) {
  for (const who of Object.keys(heads)) if (who.startsWith('rider')) delete heads[who]
  const beat = Math.floor(p.time * 8)
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  const lamp = (i: number, j: number) => {
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const r = Math.hypot(di, dj)
        if (r > 0) plot(i + di, j + dj, ink.glow, (1 - r / 2.8) * 0.6)
      }
    dot(d, g, i, j, ink.lamp)
  }

  // The lift's bars, climbing the thread while it runs.
  const { lift } = PATHS
  for (let n = Math.floor((p.lift * LIFT_V - lift.length) / (LIFT_V * BARS)); ; n++) {
    const s = p.lift * LIFT_V - n * LIFT_V * BARS
    if (s < 0) break
    if (s > lift.length) continue
    const at = along(lift, s)
    dot(d, g, col(g, at.x) + 1, row(g, at.y) - 1, ink.steel)
  }

  // The beacon on the start hut's roof: three beeps, then green as the rider goes.
  const si = col(g, MAIN.x[0])
  const sj = row(g, MAIN.y[0])
  if (p.go > 0) dot(d, g, si, sj - 6, ink.beacon[1])
  else if (p.count >= 0 && p.count % 0.6 < 0.3) dot(d, g, si, sj - 6, ink.beacon[0])
  else if (p.env.open && held(p)) dot(d, g, si, sj - 6, ink.beacon[0])

  // The photographer, crouched by the table's landing through opening hours, camera up.
  if (p.env.open) {
    const pi = col(g, PHOTO[0])
    const pj = row(g, PHOTO[1])
    dot(d, g, pi, pj - 1, ink.shorts)
    dot(d, g, pi, pj - 2, ink.jerseys[0])
    dot(d, g, pi, pj - 3, ink.skin)
    dot(d, g, pi - 1, pj - 3, p.flash > 0 ? ink.flash : ink.tyre)
    if (p.flash > 0) {
      for (const [di, dj] of [
        [-2, -3],
        [-1, -4],
        [-1, -2],
      ])
        dot(d, g, pi + di, pj + dj, ink.flash)
      if (dark) lamp(pi - 1, pj - 3)
    }
  }

  // The first-aid hut: white under a tin roof with a red cross on its side, a lamp over the door
  // when it is dark, and bikes fetched down off the line against its far wall.
  const hi = col(g, HUT[0])
  const hj = row(g, HUT[1])
  for (let dj = 1; dj <= 4; dj++)
    for (let di = -2; di <= 2; di++) {
      const cross = (di === 0 && dj >= 2) || (dj === 3 && Math.abs(di) === 1)
      dot(d, g, hi + di, hj - dj, cross ? ink.cross : ink.hut)
    }
  for (let di = -3; di <= 3; di++) dot(d, g, hi + di, hj - 5, ink.steel)
  if (dark && p.env.open) lamp(hi + 3, hj - 4)
  // The barn, with the yard light over its doors through the dark, and the farmer at it with a
  // lantern.
  paintBarn(d, g, ink.herd, p.herd)
  if (dark) lamp(col(g, BARN[0]), row(g, BARN[1]) - 4)
  paintFarmer(d, g, ink.herd, p.herd, p.time)
  paintTrough(d, g, ink.herd, p.herd)
  const farmer = p.herd.farmer
  if (dark && farmer.chore !== 'home' && farmer.chore !== 'make')
    lamp(col(g, farmer.x) + farmer.face, row(g, farmer.y) - 2)
  p.leaning.forEach((b, n) => {
    dot(d, g, hi - 3 - n, hj - 1, ink.tyre)
    dot(d, g, hi - 3 - n, hj - 2, ink.frames[b.frame % ink.frames.length])
    dot(d, g, hi - 3 - n, hj - 3, ink.tyre)
  })
  // Bikes left lying on the line after a board call. The first stays on the list until it reaches
  // the hut, so skip it while a medic wheels it down.
  const wheeling = p.medics.some(m => m.state === 'wheel')
  for (const w of p.wrecks) {
    if (wheeling && w === p.wrecks[0]) continue
    const at = onTrack(TRACKS[w.line], w.q)
    const [wi, wj] = [col(g, at.x), row(g, at.y)]
    dot(d, g, wi - 1, wj - 1, ink.tyre)
    dot(d, g, wi, wj - 1, ink.frames[w.frame % ink.frames.length])
    dot(d, g, wi + 1, wj - 1, ink.tyre)
  }

  // A medic: shorts, the patrol jersey with the kit bag on its back, and a dark helmet; knelt, the
  // jersey sits on the knee and a hand reaches out.
  const standing = (i: number, j: number, f: number, walking: boolean) => {
    if (walking) {
      dot(d, g, i - 1, j - 1, ink.shorts)
      dot(d, g, i + 1, j - 1, ink.shorts)
    } else dot(d, g, i, j - 1, ink.shorts)
    dot(d, g, i, j - 2, ink.patrol)
    dot(d, g, i, j - 3, ink.tyre)
    dot(d, g, i - f, j - 2, ink.kit)
  }
  const kneeling = (i: number, j: number, f: number) => {
    dot(d, g, i, j - 1, ink.shorts)
    dot(d, g, i, j - 2, ink.patrol)
    dot(d, g, i, j - 3, ink.tyre)
    dot(d, g, i + f, j - 2, ink.skin)
  }
  const medicHead = (m: Medic, at: Pt) => {
    heads[`rider-m${p.medics.indexOf(m)}`] = at
  }

  // The herd and the riders, nearer over farther.
  const herd = p.herd.beasts
  const beasts = [...herd].sort((a, b) => a.y - b.y)
  let nb = 0
  const pen: Pen = { d, g, ink, p, dark, beat, plot, lamp, kneeling, medicHead }
  const order = p.riders.map(r => ({ r, at: place(r) })).sort((a, b) => a.at.ground - b.at.ground)
  for (const { r, at } of order) {
    for (; nb < beasts.length && beasts[nb].y <= at.ground; nb++)
      paintBeast(d, g, ink.herd, beasts[nb], p.time, herd.indexOf(beasts[nb]))
    heads[`rider${r.id}`] = paintRider(pen, r, at)
  }
  for (; nb < beasts.length; nb++)
    paintBeast(d, g, ink.herd, beasts[nb], p.time, herd.indexOf(beasts[nb]))

  // The medics on their feet: one at the hut's door through opening hours, and whoever is out on a
  // call, the second up the line with the board on their back, the last down it wheeling the bike.
  const door = p.medics.find(m => m.state === 'post')
  for (const m of p.medics) {
    const out = m.walk && ['go', 'lead', 'back', 'fetch', 'wheel'].includes(m.state)
    if (!out && !(m === door && p.env.open)) continue
    const at = out ? walkAt(m.walk!) : { x: HUT[0] + 14, y: HUT[1], face: 1 }
    const i = col(g, at.x)
    const j = row(g, at.y)
    const f = at.face
    standing(i, j, f, !!out && Math.floor(p.time * 2.5 + p.medics.indexOf(m)) % 2 === 1)
    if (m.state === 'go' && m.board)
      for (let dj = 1; dj <= 3; dj++) dot(d, g, i - f, j - dj, ink.kit)
    const wreck = p.wrecks[0]
    if (m.state === 'wheel' && wreck) {
      dot(d, g, i + f, j - 2, ink.skin)
      dot(d, g, i + f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 1, ink.frames[wreck.frame % ink.frames.length])
      dot(d, g, i + 3 * f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 2, ink.tyre)
    }
    if (dark && out) lamp(i, j - 3)
    medicHead(m, [i, j - 3])
  }
}
