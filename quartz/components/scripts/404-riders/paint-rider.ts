import type { Medic, Park, Rider, RiderInk } from './park-state'
import { TRACKS } from '../404-downhill'
import { col, dot, row, type Grid, type Pt, type RGB } from '../404-pixel'
import { place } from './positions'

// What paintPark hands each rider: the buffer and its grid, the inks, the beat, and the dither,
// lamp and medic helpers over them.
export type Pen = {
  d: Uint8ClampedArray
  g: Grid
  ink: RiderInk
  p: Park
  dark: boolean
  beat: number
  plot: (i: number, j: number, c: RGB, alpha: number) => void
  lamp: (i: number, j: number) => void
  kneeling: (i: number, j: number, f: number) => void
  medicHead: (m: Medic, at: Pt) => void
}

// One rider in their state, with whoever tends or carries them; returns the pixel over their head.
export function paintRider(pen: Pen, r: Rider, at: ReturnType<typeof place>): [number, number] {
  const { d, g, ink, p, dark, beat, plot, lamp, kneeling, medicHead } = pen
  const jersey = ink.jerseys[r.kit[0] % ink.jerseys.length]
  const helmet = ink.helmets[r.kit[1] % ink.helmets.length]
  const frame = ink.frames[r.kit[2] % ink.frames.length]
  const f = at.face
  const i = col(g, at.x)
  const j = row(g, at.y)
  const tall = r.kid ? 3 : 4
  let head: [number, number] = [i, j - tall]
  const moving = !['gate', 'liftq', 'down', 'hurt', 'carried', 'rest'].includes(r.state)
  switch (r.state) {
    case 'ride':
    case 'air':
    case 'runout':
    case 'leave': {
      // On the bike, weight back, bars forward. A foot down in a slide; a pixel of compression
      // off a heavy landing; the odd root jolting them off the saddle.
      const sink = r.heavy > 0 ? 1 : 0
      const jolt =
        r.state === 'ride' && !sink && Math.sin(r.q * 0.37 + r.id) > 0.93 && r.v > 40 ? 1 : 0
      const y0 = j - 1
      const b = y0 - 1 + sink - jolt
      const whip = r.trick === 'whip'
      dot(d, g, i - f, whip ? y0 - 1 : y0, ink.tyre)
      dot(d, g, i, y0, frame)
      dot(d, g, i + f, y0, ink.tyre)
      if (r.trick === 'superman') {
        // Stretched out flat behind the bars.
        dot(d, g, i + f, b, ink.tyre)
        dot(d, g, i, b, jersey)
        dot(d, g, i - f, b, ink.shorts)
        dot(d, g, i - 2 * f, b, ink.shorts)
        dot(d, g, i + f, b - 1, helmet)
        head = [i + f, b - 1]
      } else {
        if (r.trick !== 'nohander') dot(d, g, i + f, b, ink.tyre)
        if (r.kid) {
          dot(d, g, i, b, jersey)
          dot(d, g, i - f, b - 1, helmet)
          head = [i - f, b - 1]
        } else {
          dot(d, g, i, b, ink.shorts)
          dot(d, g, i - f, b - 1, jersey)
          dot(d, g, i, b - 1, jersey)
          dot(d, g, i - f, b - 2, helmet)
          head = [i - f, b - 2]
        }
        if (r.trick === 'nohander') {
          dot(d, g, head[0] - f, head[1] + 1, ink.skin)
          dot(d, g, head[0] + 2 * f, head[1] + 1, ink.skin)
        }
      }
      if (whip) dot(d, g, i - 2 * f, y0 - 1, frame)
      if (r.dab > 0) dot(d, g, i + f * (beat & 1 ? 1 : 0), y0 + 1, ink.shorts)
      if (r.state === 'air') dot(d, g, i, row(g, at.ground), ink.dust)
      else if (r.state === 'ride') {
        const t = TRACKS[r.line]
        // Roost off the back wheel out of a berm, dust on the straights, a puff off a landing.
        if (t.bank[r.k] > 0.5 && r.v > 50) {
          dot(d, g, i - 2 * f, y0 - (beat & 1), ink.dust)
          dot(d, g, i - 2 * f, y0 - 1 + Math.sign(t.bend[r.k]) * f, ink.dust)
          if (beat & 1) dot(d, g, i - 3 * f, y0 - 1, ink.dust)
        } else if (r.v > 40 && beat & 1) dot(d, g, i - 2 * f, y0, ink.dust)
        if (sink) {
          dot(d, g, i - 2, y0 + 1, ink.dust)
          dot(d, g, i + 2, y0 + 1, ink.dust)
        }
      }
      break
    }
    case 'down':
    case 'hurt': {
      // The bike on its side where it stopped, the rider tumbling past it and then lying there.
      dot(d, g, i - 1, j - 1, ink.tyre)
      dot(d, g, i, j - 1, frame)
      dot(d, g, i + 1, j - 1, ink.tyre)
      const r0 = i + 2 * f
      const fresh = r.state === 'down'
      if (fresh && r.t < 0.35 && beat & 1) {
        dot(d, g, r0, j - 1, helmet)
        dot(d, g, r0, j - 2, jersey)
        dot(d, g, r0, j - 3, ink.shorts)
        head = [r0, j - 3]
      } else {
        dot(d, g, r0, j - 1, ink.shorts)
        dot(d, g, r0 + f, j - 1, jersey)
        dot(d, g, r0 + 2 * f, j - 1, helmet)
        head = [r0 + 2 * f, j - 2]
      }
      if (fresh && r.t < 0.6) {
        plot(i - f, j - 2, ink.dust, 0.9)
        plot(i + 3 * f, j - 3, ink.dust, 0.6)
        plot(i, j - 3, ink.dust, 0.7)
      }
      // A medic knelt at their head with the bag beside; on a board call the second knelt across
      // the bike with the board slid under them.
      for (const m of p.medics) {
        if (m.who !== r || m.state !== 'tend') continue
        if (m.board) {
          kneeling(i - 2 * f, j, f)
          for (let di = 2; di <= 4; di++) dot(d, g, i + di * f, j, ink.kit)
          medicHead(m, [i - 2 * f, j - 3])
        } else {
          kneeling(i + 5 * f, j, -f)
          dot(d, g, i + 6 * f, j - 1, ink.kit)
          medicHead(m, [i + 5 * f, j - 3])
        }
      }
      break
    }
    case 'carried': {
      // Feet first down the hill on the board, a medic at either end.
      for (let di = -2; di <= 2; di++) dot(d, g, i + di, j - 2, ink.kit)
      dot(d, g, i + f, j - 3, ink.shorts)
      dot(d, g, i, j - 3, jersey)
      dot(d, g, i - f, j - 3, helmet)
      head = [i - f, j - 3]
      const step = Math.floor(p.time * 2.2) % 2
      p.medics
        .filter(m => m.state === 'carry')
        .forEach((m, n) => {
          const e = n ? -1 : 1
          const at: Pt = [i + 3 * e * f, j]
          if ((step + n) % 2) {
            dot(d, g, at[0] - 1, j - 1, ink.shorts)
            dot(d, g, at[0] + 1, j - 1, ink.shorts)
          } else dot(d, g, at[0], j - 1, ink.shorts)
          dot(d, g, at[0], j - 2, ink.patrol)
          dot(d, g, at[0], j - 3, ink.tyre)
          if (dark) lamp(at[0], j - 3)
          medicHead(m, [at[0], j - 3])
        })
      break
    }
    case 'rest': {
      // Sat on the grass by the hut with the helmet off beside them, an arm in a white sling, and
      // the bike laid down.
      dot(d, g, i, j - 1, ink.shorts)
      dot(d, g, i + 1, j - 1, ink.shorts)
      dot(d, g, i, j - 2, jersey)
      dot(d, g, i + 1, j - 2, ink.hut)
      dot(d, g, i, j - 3, ink.skin)
      dot(d, g, i - 1, j - 1, helmet)
      dot(d, g, i + 3, j - 1, ink.tyre)
      dot(d, g, i + 4, j - 1, frame)
      dot(d, g, i + 5, j - 1, ink.tyre)
      head = [i, j - 3]
      break
    }
    case 'lift': {
      // Sat on the bike, one hand up on the bar.
      dot(d, g, i - 1, j - 1, ink.tyre)
      dot(d, g, i, j - 1, frame)
      dot(d, g, i + 1, j - 1, ink.tyre)
      dot(d, g, i, j - 2, ink.shorts)
      dot(d, g, i, j - 3, jersey)
      dot(d, g, i, j - 4, helmet)
      dot(d, g, i + 1, j - 4, ink.skin)
      dot(d, g, i + 1, j - 5, ink.steel)
      head = [i, j - 4]
      break
    }
    case 'gate': {
      // Astride the bike, waiting their turn.
      dot(d, g, i - 1, j - 1, ink.tyre)
      dot(d, g, i, j - 1, frame)
      dot(d, g, i + 1, j - 1, ink.tyre)
      if (!r.kid) dot(d, g, i, j - 2, ink.shorts)
      dot(d, g, i, j - tall + 1, jersey)
      dot(d, g, i, j - tall, helmet)
      break
    }
    default: {
      // On foot beside the bike, a hand on the bars, walking or standing: arriving, queueing,
      // pushing up the lift track, or getting up after a fall.
      const walking =
        (r.state === 'arrive' || r.state === 'hike' || r.state === 'limp') &&
        Math.floor(p.time * (r.state === 'limp' ? 1.5 : 2.5) + r.id) % 2
      if (walking) {
        dot(d, g, i - 1, j - 1, ink.shorts)
        dot(d, g, i + 1, j - 1, ink.shorts)
      } else dot(d, g, i, j - 1, ink.shorts)
      dot(d, g, i, j - 2, r.kid ? helmet : jersey)
      if (!r.kid) dot(d, g, i, j - 3, helmet)
      dot(d, g, i + f, j - 2, ink.skin)
      dot(d, g, i + f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 1, frame)
      dot(d, g, i + 3 * f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 2, ink.tyre)
      head = [i, j - (r.kid ? 2 : 3)]
    }
  }
  if (dark && moving) lamp(head[0], head[1])
  return head
}
