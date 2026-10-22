import type { Voices } from '../404-crew'
import { dot, type Grid, type Pt, type RGB } from '../404-pixel'
import {
  clamp,
  LIFT_LEN,
  LIGHT,
  onRun,
  RUNS,
  spot,
  THROW,
  ZONES,
  type Hill,
  type Skier,
  type SkiInk,
} from './hill-state'

export function paintSkiers(scope: {
  d: Uint8ClampedArray
  g: Grid
  ink: SkiInk
  h: Hill
  onRange: (i: number, j: number) => boolean
  dark: boolean
  heads: Voices['heads']
  plot: (i: number, j: number, c: RGB, alpha: number) => void
  at: ([x, y]: Pt) => Pt
  beat: number
  lift: (u: number) => Pt
}) {
  const { d, g, ink, h, onRange, dark, heads, plot, at, beat, lift } = scope
  const where = (o: Skier): Pt => {
    if (o.state === 'chair') {
      const u = clamp((h.lift - o.chair) / LIFT_LEN, 0, 1)
      return lift(u)
    }
    if (onRun(o) || o.state === 'collect') return spot(RUNS[o.run], o.s, o.w)
    if (o.state === 'out') return spot(RUNS[o.run], RUNS[o.run].length - 4, o.w)
    return [o.x, o.y]
  }
  const coatOf = (o: Skier) =>
    o.role === 'teach'
      ? ink.school
      : o.role === 'patrol'
        ? ink.patrol
        : ink.coats[o.kit[0] % ink.coats.length]
  const helmetOf = (o: Skier) =>
    o.role === 'patrol' ? ink.boot : ink.helmets[o.kit[1] % ink.helmets.length]
  // The patrol's toboggan is a little over two metres long, under a pixel at this range, so it is
  // drawn to the scale of the people: on the snow behind the patroller in the handles, with whoever
  // is strapped in lying on it feet first.
  const toboggan = (i: number, j: number, side: number, c: Skier | null) => {
    dot(d, g, i + side, j - 1, ink.steel)
    for (let n = 2; n <= 4; n++) dot(d, g, i + side * n, j, ink.sled)
    if (!c) return
    dot(d, g, i + side * 2, j - 1, ink.pants)
    dot(d, g, i + side * 3, j - 1, coatOf(c))
    dot(d, g, i + side * 4, j - 1, helmetOf(c))
  }
  const order = h.skiers.map(o => ({ o, p: at(where(o)) })).sort((a, b) => a.p[1] - b.p[1])
  for (const { o, p } of order) {
    let [i, j] = p
    if (o.state === 'chair') i += o.seat
    if (!onRange(i, j - 1) && o.state !== 'chair') continue
    // Strapped into the toboggan, drawn with the patroller.
    if (o.role !== 'patrol' && (o.state === 'sled' || o.state === 'haul')) continue
    const coat = coatOf(o)
    const helmet = helmetOf(o)
    const gear = o.board
      ? ink.boards[o.kit[2] % ink.boards.length]
      : ink.gear[o.kit[2] % ink.gear.length]
    const tall = o.kid ? 3 : 4
    let head: Pt = [i, j - tall + 1]
    const face = Math.sin(o.psi) >= 0 ? 1 : -1
    switch (o.state) {
      case 'chair':
        // Sat on the chair, skis or board hanging below.
        dot(d, g, i, j, coat)
        dot(d, g, i, j - 1, helmet)
        dot(d, g, i, j + 2, gear)
        head = [i, j - 1]
        break
      case 'ski':
      case 'ready':
      case 'out': {
        const lean =
          o.mode === 'leaf' || o.mode === 'plough' ? 0 : Math.abs(o.psi) < o.amp * 0.8 ? o.dir : 0
        if (o.mode === 'plough') {
          // The wedge, tips together: two feet apart under the body.
          dot(d, g, i - 1, j, gear)
          dot(d, g, i + 1, j, gear)
        } else for (let di = -1; di <= 1; di++) dot(d, g, i + di, j, gear)
        if (o.kid) {
          dot(d, g, i, j - 1, coat)
          dot(d, g, i + lean, j - 2, helmet)
          head = [i + lean, j - 2]
        } else {
          const tuck = o.mode === 'race' && Math.abs(o.psi) < 0.3
          dot(d, g, i, j - 1, ink.pants)
          dot(d, g, i + lean, j - 2, coat)
          if (tuck) {
            dot(d, g, i + lean + face, j - 2, helmet)
            head = [i + lean + face, j - 2]
          } else {
            dot(d, g, i + lean, j - 3, helmet)
            head = [i + lean, j - 3]
          }
          // Arms out for balance sliding sideways on a board; a boarder stands side-on with the
          // leading arm down the board, a skier plants a pole.
          if (o.mode === 'leaf') {
            dot(d, g, i - 1, j - 2, ink.skin)
            dot(d, g, i + 1, j - 2, ink.skin)
          } else if (o.board) dot(d, g, i + lean + face, j - 2, coat)
          else dot(d, g, i - face, j - 1, ink.steel)
        }
        // Snow off the outside of a skidded turn or a stop.
        const spray = o.state === 'out' ? 1 : o.v > 3 ? o.skid * Math.abs(Math.sin(o.psi)) : 0
        for (let n = 1; n <= 3; n++)
          if (spray * 1.2 > n / 3 && (beat + n) % 2)
            dot(d, g, i + face * (1 + n), j - (n > 1 ? 1 : 0), ink.snow[3])
        break
      }
      case 'down':
      case 'up':
        if (o.state === 'down' || o.t < 1) {
          // Flat out on the snow.
          dot(d, g, i - 1, j - 1, helmet)
          dot(d, g, i, j - 1, coat)
          dot(d, g, i + 1, j - 1, ink.pants)
          dot(d, g, i + 2, j, o.gear >= 0 ? ink.boot : gear)
          head = [i - 1, j - 1]
        } else {
          dot(d, g, i, j - 1, coat)
          dot(d, g, i, j - 2, helmet)
          dot(d, g, i + 1, j, gear)
          head = [i, j - 2]
        }
        break
      case 'skin': {
        // Skinning up, upright, a pole planted on each stride.
        const stride = Math.floor(h.time * 1.5 + o.id) % 2
        dot(d, g, i, j, gear)
        dot(d, g, i, j + 1 - stride, gear)
        dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - 2, coat)
        dot(d, g, i, j - 3, helmet)
        dot(d, g, i - 1, j - 1 - stride, ink.steel)
        dot(d, g, i + 1, j - 2 + stride, ink.steel)
        head = [i, j - 3]
        break
      }
      case 'hurt': {
        // Flat out where they stopped, and above them their skis crossed in the snow, or the board
        // stood on its tail, so that nobody comes over the rise into them.
        dot(d, g, i - 1, j - 1, helmet)
        dot(d, g, i, j - 1, coat)
        dot(d, g, i + 1, j - 1, ink.pants)
        head = [i - 1, j - 1]
        if (o.board) for (let n = 3; n < 6; n++) dot(d, g, i, j - n, gear)
        else
          for (const [di, dj] of [
            [-1, -5],
            [1, -5],
            [0, -4],
            [-1, -3],
            [1, -3],
          ])
            dot(d, g, i + di, j + dj, gear)
        break
      }
      case 'rescue':
      case 'sled':
      case 'haul': {
        // In the handles: down to them with the toboggan empty, down to the lift's foot with them in
        // it, and walking it in to the shed. It goes on the side it trails to.
        const c = o.state === 'rescue' ? null : o.patient
        const tow = c ? where(c) : spot(RUNS[o.run], Math.max(0, o.s - 2.5), o.w)
        dot(d, g, i, j, o.state === 'haul' ? ink.boot : gear)
        dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - 2, coat)
        dot(d, g, i, j - 3, helmet)
        head = [i, j - 3]
        toboggan(i, j, Math.sign(tow[0] - where(o)[0]) || 1, c)
        break
      }
      case 'charge': {
        // Stood under the crest pulling the igniter, the charge thrown out over the slope, then back
        // behind the rib to wait for it.
        dot(d, g, i, j, ink.boot)
        dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - 2, coat)
        dot(d, g, i, j - 3, helmet)
        head = [i, j - 3]
        const z = ZONES[o.shot]
        const side = Math.sign(z.lands[0] - z.stand[0]) || 1
        if (o.t >= LIGHT && o.t < THROW && beat % 2) dot(d, g, i + side, j - 2, ink.lamp)
        if (o.t >= THROW && o.t < THROW + 1) {
          const u = o.t - THROW
          dot(d, g, i + side, j - 4, ink.skin)
          dot(
            d,
            g,
            ...at([
              z.stand[0] + (z.lands[0] - z.stand[0]) * u,
              z.stand[1] + (z.lands[1] - z.stand[1]) * u - 16 * u * (1 - u),
            ]),
            ink.boot,
          )
        }
        break
      }
      case 'tend': {
        // Knelt beside them on the side they skied in from, the toboggan laid out beyond.
        if (!o.patient) break
        const [ki, kj] = at(where(o.patient))
        const side = Math.sign(o.w - o.patient.w) || 1
        const ti = ki + 2 * side
        dot(d, g, ti, kj, ink.boot)
        dot(d, g, ti, kj - 1, coat)
        dot(d, g, ti, kj - 2, helmet)
        head = [ti, kj - 2]
        for (let n = 2; n <= 4; n++) dot(d, g, ti + side * n, kj, ink.sled)
        break
      }
      case 'strap':
        // Sat in the snow at the top, strapping in.
        for (let di = -1; di <= 1; di++) dot(d, g, i + di, j, gear)
        dot(d, g, i, j - 1, coat)
        dot(d, g, i, j - 2, helmet)
        head = [i, j - 2]
        break
      case 'coach':
        dot(d, g, i, j, ink.pants)
        dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - 2, coat)
        dot(d, g, i, j - 3, helmet)
        if (h.cheer > 0) {
          dot(d, g, i - 1, j - 3 - (beat & 1), ink.skin)
          dot(d, g, i + 1, j - 4 + (beat & 1), ink.skin)
        }
        head = [i, j - 3]
        break
      default: {
        // On foot: stood in the queue or walking, skis on the shoulder or the board under an arm;
        // or sidestepping back up to lost skis.
        const stride = o.state !== 'queue' && Math.floor(h.time * 2.5 + o.id) % 2
        if (stride) {
          dot(d, g, i - 1, j, ink.boot)
          dot(d, g, i + 1, j, ink.boot)
        } else dot(d, g, i, j, ink.boot)
        if (!o.kid) dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - tall + 2, coat)
        dot(d, g, i, j - tall + 1, helmet)
        if (o.board) dot(d, g, i + 1, j - 1, gear)
        else if (o.state !== 'collect') {
          dot(d, g, i + 1, j - tall + 1, gear)
          dot(d, g, i + 1, j - tall, gear)
        }
      }
    }
    // Uphillers come in, climb, ride down and go home by headlamp, and the patrol wears one once
    // the floodlights are off.
    if (dark && (o.role === 'tour' || (o.role === 'patrol' && !h.env.open))) {
      const [hi, hj] = head
      for (let dj = -2; dj <= 2; dj++)
        for (let di = -2; di <= 2; di++) {
          const r = Math.hypot(di, dj)
          if (r > 0) plot(hi + di, hj + dj, ink.glow, (1 - r / 2.8) * 0.6)
        }
      dot(d, g, hi, hj, ink.lamp)
    }
    // Skis and poles where they came off.
    if (o.gear >= 0) {
      const [gi, gj] = at(spot(RUNS[o.run], o.gear, o.gearW))
      dot(d, g, gi - 1, gj, gear)
      dot(d, g, gi + 1, gj - 1, gear)
      dot(d, g, gi, gj - 1, ink.steel)
    }
    heads[`ski${o.id}`] = head
  }
}
