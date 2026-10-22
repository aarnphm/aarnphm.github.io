import { say, type Voices } from '../404-crew'
import {
  BOOM,
  cell,
  clamp,
  FINISH,
  free,
  G,
  GAP,
  lane,
  laneW,
  LIFT_HEAD,
  LIFT_LEN,
  LIFT_V,
  mark,
  QUEUE,
  RUNS,
  seek,
  spot,
  spotW,
  THROW,
  WAY,
  ZONES,
  type Hill,
  type Skier,
} from './hill-state'
import {
  camp,
  control,
  descend,
  finish,
  patrol,
  recruit,
  rescue,
  skin,
  tourers,
  walk,
} from './population'
import { ski, tumble } from './ski-motion'

const byTicket = (a: Skier, b: Skier) => a.ticket - b.ticket

const skiQueue: Skier[] = []

export function stepHill(h: Hill, dt: number, voices: Voices) {
  h.time += dt
  h.cheer = Math.max(0, h.cheer - dt)
  for (let n = 0; n < h.flick.length; n++) h.flick[n] = Math.max(0, h.flick[n] - dt)
  const env = h.env

  // The chair runs through its hours, and on until whoever is on it is off; the wind stops it.
  const riding = h.skiers.some(o => o.state === 'chair')
  if (!env.hold && (env.open || riding)) h.lift += LIFT_V * dt
  const slot = Math.floor(h.lift / GAP)
  const board = env.open && slot !== h.loaded
  h.loaded = slot

  if (env.camp && !h.skiers.some(o => o.role === 'coach')) camp(h)
  if (env.control && !h.skiers.some(o => o.shot >= 0)) control(h)
  let live = 0
  let touring = 0
  skiQueue.length = 0
  for (let i = 0; i < h.skiers.length; i++) {
    const o = h.skiers[i]
    if (free(o)) live++
    if (o.role === 'tour' && o.state !== 'leave') touring++
    if (o.state === 'queue') skiQueue.push(o)
  }
  h.arrive = Math.max(0, h.arrive - dt)
  if (!h.arrive && live < env.crowd) {
    recruit(h, env.crowd - live)
    h.arrive = 8 + h.rng() * 20
  } else if (!h.arrive && touring < env.tour) {
    tourers(h)
    h.arrive = 40 + h.rng() * 150
  }

  // Snow falling fills the tracks in.
  if ((h.fill += (env.fresh * dt) / 30) >= 1) {
    h.fill -= 1
    for (const t of h.tracks) for (let c = 0; c < t.length; c++) if (t[c] > 1) t[c]--
  }
  stepCat(h, dt)

  const queue = skiQueue.sort(byTicket)
  // A charge is seen at the foot a second before it is heard, some four hundred metres off, and the
  // line for first chair cheers it.
  let bw = 0
  for (let i = 0; i < h.blasts.length; i++) {
    const b = h.blasts[i]
    if (b.t < 1.1 && b.t + dt >= 1.1)
      for (const o of queue) if (h.rng() < 0.25) say(voices, `ski${o.id}`, '!')
    b.t += dt
    // A slide's crown and bed stay on the face for the rest of the morning; smoke alone is soon gone.
    if (b.t < (b.slide ? 6 * 3600 : 40)) h.blasts[bw++] = b
  }
  h.blasts.length = bw
  const steps = Math.ceil(dt * 30)
  for (const o of h.skiers) {
    o.t += dt
    switch (o.state) {
      case 'arrive':
        if (!walk(o, dt)) break
        if (o.role === 'tour') skin(o)
        else {
          o.state = 'walk'
          o.path = [QUEUE(queue.length)]
        }
        break
      case 'skin': {
        // 0.6 to 0.75 m/s up the slope, six to seven hundred vertical metres an hour on the blue's
        // average of 16°: a fit local's pace on groomed snow, the same all the way up.
        const run = RUNS[0]
        o.k = seek(run, o.s, o.k)
        // The one in front eases off for their partner.
        const apart = h.skiers.some(
          p =>
            p !== o &&
            p.party === o.party &&
            (p.state === 'skin' || p.state === 'arrive') &&
            p.s > o.s + 8,
        )
        o.s = Math.max(0, o.s - (0.45 + 0.3 * o.skill) * (apart ? 0.5 : 1) * dt)
        o.w = lane(run, o.k)
        mark(h, o, run)
        if (o.s <= 0) {
          o.state = 'strap'
          ;[o.x, o.y] = spot(run, 0, o.w)
          o.t = 0
        }
        break
      }
      case 'walk':
        if (o.path.length) walk(o, dt)
        else if (o.shot >= 0) {
          // The patrol boards past the line.
          o.state = 'chair'
          o.chair = Math.floor(h.lift / GAP) * GAP
          o.seat = 0
          o.shot = 0
        } else {
          o.state = 'queue'
          o.ticket = ++h.tickets
          queue.push(o)
        }
        break
      case 'queue': {
        const n = queue.indexOf(o)
        o.path = [QUEUE(n)]
        walk(o, dt)
        if (!env.open && !env.hold && !env.control) {
          o.state = 'leave'
          o.path = WAY.slice().reverse()
        }
        break
      }
      case 'chair':
        if (h.lift - o.chair >= LIFT_LEN && o.role === 'patrol') {
          o.x = LIFT_HEAD[0]
          o.y = LIFT_HEAD[1]
          if (o.shot >= 0 && env.control) {
            o.state = 'route'
            o.path = [ZONES[o.shot].stand]
          } else if (o.shot >= 0) descend(o)
          else {
            o.state = 'leave'
            o.path = [[LIFT_HEAD[0] + 5, LIFT_HEAD[1] + 1]]
          }
        } else if (h.lift - o.chair >= LIFT_LEN) {
          o.state = 'top'
          o.x = LIFT_HEAD[0]
          o.y = LIFT_HEAD[1]
          o.run =
            o.role === 'free' && o.mode !== 'plough' && o.mode !== 'leaf' && choose(h, o) ? 1 : 0
          const [x, y] = spot(RUNS[o.run], 0, o.board ? -RUNS[o.run].half[0] * 0.6 : 0)
          o.path = [[x, y]]
          o.t = 0
        }
        break
      case 'top':
        if (walk(o, dt * 1.5)) {
          o.state = o.board ? 'strap' : 'ready'
          o.t = 0
        }
        break
      case 'strap':
        // Strapping in; at the top of a climb, skins off and the splitboard back together.
        if (o.t > (o.role === 'tour' ? 60 + 60 * (1 - o.skill) : 5 + 10 * (1 - o.skill))) {
          o.state = 'ready'
          o.t = 0
        }
        break
      case 'ready':
        drop(h, o)
        break
      case 'ski':
        for (let n = 0; n < steps && o.state === 'ski'; n++) ski(h, o, dt / steps, voices)
        break
      case 'down':
      case 'up':
      case 'collect':
        tumble(o, dt)
        break
      case 'hurt': {
        const by = h.skiers.some(p => p.asked === -o.id && p.stop > 0 && Math.abs(p.s - o.s) < 20)
        if ((by || o.t > 30) && !h.skiers.some(p => p.patient === o)) patrol(h, o)
        break
      }
      case 'rescue':
      case 'tend':
        rescue(h, o, dt, voices)
        break
      case 'sled':
      case 'haul':
        if (o.role === 'patrol') rescue(h, o, dt, voices)
        break
      case 'route':
        // Along under the crest on foot, skis on the shoulder; once the chair opens, down.
        if (!env.control) descend(o)
        else if (walk(o, dt)) {
          o.state = 'charge'
          o.t = 0
        }
        break
      case 'charge': {
        const z = ZONES[o.shot]
        if (o.t >= THROW && o.t - dt < THROW) o.path = [z.hide]
        walk(o, dt)
        if (o.t >= BOOM && o.t - dt < BOOM)
          h.blasts.push({
            x: z.lands[0],
            y: z.lands[1],
            t: 0,
            slide: h.rng() < 0.55 ? 20 + h.rng() * 50 : 0,
          })
        // A look at what it did, then the next zone, or down once the last is shot.
        if (o.t >= BOOM + 15) {
          if (env.control && o.shot < ZONES.length - 1) {
            o.shot++
            o.state = 'route'
            o.path = [ZONES[o.shot].stand]
          } else descend(o)
        }
        break
      }
      case 'out':
        // A hockey stop.
        o.v = Math.max(0, o.v - 0.7 * G * dt)
        if (o.v === 0) finish(h, o)
        break
      case 'leave':
        if (walk(o, dt)) o.gone = true
        break
      case 'coach':
        if (!env.camp && !h.skiers.some(p => p.role === 'racer' && p.state !== 'leave')) {
          o.state = 'leave'
          o.path = [spot(RUNS[0], RUNS[0].length - 3, 0), [1322, 452], ...WAY.slice().reverse()]
        }
        break
    }
  }

  // A chair at the foot: the head of the queue sits down, and whoever is next with them if they
  // came together, or if both came alone.
  if (board && queue.length) {
    const first = queue[0]
    const near = Math.hypot(first.x - QUEUE(0)[0], first.y - QUEUE(0)[1]) < 1
    if (near) {
      const alone = (o: Skier) => !queue.some(p => p !== o && p.party === o.party)
      const second = queue[1]
      const pair =
        second && (second.party === first.party || (alone(first) && alone(second))) ? second : null
      for (const [seat, o] of [first, pair].entries()) {
        if (!o) continue
        o.state = 'chair'
        o.chair = slot * GAP
        o.seat = seat
        o.t = 0
      }
    }
  }
  let sw = 0
  for (let i = 0; i < h.skiers.length; i++) {
    if (!h.skiers[i].gone) h.skiers[sw++] = h.skiers[i]
  }
  h.skiers.length = sw
}

// The lift line or the blue: the steeper the better they are and the surer of themselves.
function choose(h: Hill, o: Skier) {
  const bold = o.conf * (0.5 + o.skill) + (h.rng() - 0.5) * 0.2
  return o.skill > 0.6 && h.rng() < clamp((bold - 0.7) * 2, 0, 0.6)
}

// At the top they wait for the rest of their party (up to a minute), then drop in one after another,
// a few seconds apart; a racer waits until the course below is clear.
function drop(h: Hill, o: Skier) {
  let behind = false
  let order = 0
  let seenSelf = false
  for (let i = 0; i < h.skiers.length; i++) {
    const p = h.skiers[i]
    if (p.party !== o.party || p.role === 'coach') continue
    if (p.state === 'chair' || p.state === 'top' || p.state === 'strap' || p.state === 'skin') {
      behind = true
    }
    if (p === o) seenSelf = true
    else if (!seenSelf && (p.state === 'ready' || p.state === 'ski')) order++
  }
  if (behind && o.t < 60) return
  if (o.role === 'racer') {
    const busy = h.skiers.some(p => p.role === 'racer' && p.state === 'ski' && p.s < FINISH)
    if (busy || o.t < 4) return
  } else if (o.role !== 'pupil' && o.t < 2 + 4 * order) return
  if (o.role === 'pupil' && o.lead && o.lead.state !== 'ski' && o.t < 60) return
  start(h, o)
}

export function start(h: Hill, o: Skier) {
  o.state = 'ski'
  o.t = 0
  o.k = 0
  o.s = 0
  if (o.role !== 'tour') o.w = spotW(o)
  o.v = 0
  o.dir = h.rng() < 0.5 ? -1 : 1
  o.psi = o.dir * 0.6
  o.amp = o.mode === 'plough' ? 0.5 : 0.9
  o.skid = 0.3
  o.hurt = -1
  o.asked = 0
  o.cell = -1
  // A fresh track for the class to follow, starting where they stand.
  o.trail.fill(0)
}

// The groomer: up the blue with its blade raised, then down it combing a lane flat, lane by lane
// across the run, from after the lifts close until late, and again before dawn.
function stepCat(h: Hill, dt: number) {
  const c = h.cat
  const run = RUNS[0]
  if (!h.env.groom) {
    c.s = run.length
    c.dir = -1
    return
  }
  c.s += c.dir * (c.dir < 0 ? 2 : 3) * dt
  if (c.s <= 0) c.dir = 1
  if (c.s >= run.length) {
    c.dir = -1
    c.lane = (c.lane + 1) % 3
  }
  c.s = clamp(c.s, 0, run.length)
  if (c.dir > 0) {
    const k = seek(run, c.s, 0)
    const w = laneW(c.lane, run.half[k])
    for (let d = -2.5; d <= 2.5; d += 1) {
      const n = cell(run, c.s, w + d, k)
      h.tracks[0][n] = 0
      h.last[0][n] = -99
    }
  }
}
