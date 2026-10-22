import { say, type Voices } from '../404-crew'
import {
  clamp,
  FINISH,
  G,
  GATES,
  mark,
  onRun,
  RUNS,
  seek,
  type Hill,
  type Skier,
} from './hill-state'

// Air drag over mass, ½ρC_dA/m, for a 75 kg skier standing up in the thin air at altitude.
const DRAG = (0.5 * 0.9 * 0.5) / 75
// A body sliding on snow after a fall.
const SLIDE = 0.55
// Carving radius of a ski's sidecut and of a board's.
const CARVE = { ski: 13, board: 7.5 }

function fall(h: Hill, o: Skier, voices: Voices) {
  // A bad one, likelier the faster it was, keeps them down for the patrol: `hurt` above 1.
  const bad = o.v > 6 && h.rng() < 0.08 + 0.12 * clamp((o.v - 6) / 8, 0, 1)
  o.state = 'down'
  o.t = 0
  o.hurt = bad ? 1 + h.rng() : h.rng()
  o.conf = clamp(o.conf - 0.2, 0.05, 1)
  o.pace *= 0.85
  // Off the skis at speed, the skis and poles stay where it happened.
  if (!o.board && !bad && o.v > 7 && h.rng() < 0.6) {
    o.gear = o.s
    o.gearW = o.w
  }
  say(voices, `ski${o.id}`, '!')
  // Whoever is coming down behind stops beside them to ask.
  let near: Skier | null = null
  for (const p of h.skiers)
    if (p !== o && p.state === 'ski' && p.run === o.run && p.s < o.s && o.s - p.s < 50)
      if (!near || p.s > near.s) near = p
  if (near && near.asked !== o.id) {
    near.asked = o.id
    // After a bad one they stay with them until the patrol comes.
    near.stop = bad ? 600 : 6
  }
}

// Down the run. Gravity along the heading against the ski's glide and the air; turns carved round
// the sidecut, or pivoted and skidded, which bleeds speed; and the speed held down by how far
// across the hill each turn finishes, a little further each time it runs over what they are happy
// with. A snowplough brakes with its wedge; a beginner on a board slides down on the heel edge
// from one side of the run to the other without ever pointing down it; a racer turns round gates.
export function ski(h: Hill, o: Skier, dt: number, voices: Voices) {
  const run = RUNS[o.run]
  o.k = seek(run, o.s, o.k)
  const a = run.pitch[o.k]
  const half = run.half[o.k]
  const down = G * Math.sin(a)
  const glide = h.env.glide * G * Math.cos(a)
  // Steeper than a blue, everyone skis it slower than they would like to.
  let want = o.stop > 0 ? 0 : o.pace * (1 - Math.max(0, a - 0.4) * 1.6)
  o.stop = Math.max(0, o.stop - dt)
  if (o.stop > 0 && o.v < 0.6 && o.asked > 0) {
    say(voices, `ski${o.id}`, '?')
    o.asked = -o.asked
  }

  if (o.role === 'racer' && o.s > FINISH) want = Math.min(want, 8)

  // Keep clear of whoever is below: they have the right of way.
  let turnAway = 0
  for (const p of h.skiers) {
    if (p === o || p.run !== o.run || !onRun(p) || p.party === o.party) continue
    const ds = p.s - o.s
    const dw = p.w - o.w
    if (ds > 0 && ds < 14 && Math.abs(dw) < 3.5) {
      if (Math.sign(dw) === Math.sign(Math.sin(o.psi)) || Math.abs(o.psi) < 0.3)
        turnAway = -Math.sign(dw) || 1
      if (ds < 4 && Math.abs(dw) < 2) want = Math.min(want, p.v * 0.8)
    }
  }

  let brake = 0
  let over = 0
  if (o.mode === 'leaf') {
    // Across the run on the heel edge, sliding down as fast as they let it.
    o.psi = o.dir * 1.35
    const edge = Math.abs(o.w) > half - 1.5 && Math.sign(o.w) === o.dir
    if (edge || (turnAway && turnAway !== o.dir)) o.dir = -o.dir
    const slip = clamp(1.5 + 1.2 * o.conf, 1.2, 2.6)
    o.v = Math.hypot(slip, 1.8)
    o.s += slip * dt
    o.w += o.dir * 1.8 * dt
  } else {
    const gate = o.mode === 'race' && o.run === 0 ? GATES.find(g => g.s > o.s + 1) : undefined
    if (gate) {
      // Round the outside of the next gate's pole, the line wider the faster they come at it and
      // the edge set harder to bleed what they cannot hold. Past the finish they ski back easily.
      const wide = gate.w + Math.sign(gate.w) * (1.2 + 0.3 * clamp(o.v - want, 0, 4))
      const aim = Math.atan2(wide - o.w, gate.s - o.s)
      const rate = Math.max(o.v / 18, 0.9)
      o.psi += clamp(aim - o.psi, -rate * dt, rate * dt)
      o.skid = o.v > want ? 1 : 0.1
    } else if (o.lead && o.lead.trail.length) {
      // A pupil skis where the instructor skied, a turn or so behind them.
      const tr = o.lead.trail
      const n = tr.length / 2
      const lag = Math.min(n - 1, o.slot * 18)
      const at = (((Math.floor(h.time * 4) - lag) % n) + n) % n
      // Once the instructor is down, the rest of the way is theirs to find.
      const done = o.lead.state !== 'ski'
      const ts = done ? run.length : tr[2 * at]
      const tw = done ? 0 : tr[2 * at + 1]
      const gap = Math.hypot(ts - o.s, tw - o.w) * (ts < o.s - 1 ? -1 : 1)
      const aim = clamp(Math.atan2(tw - o.w, Math.max(1, ts - o.s)), -1.2, 1.2)
      o.psi += clamp(aim - o.psi, -0.8 * dt, 0.8 * dt)
      want = clamp(o.lead.v + 0.5 * gap, 0, 4.5)
    } else {
      // Finishing each turn further across the hill as the speed runs on past what they want.
      const target = Math.max(0.5, want)
      // An instructor leads the class in long traverses from side to side, holding the speed down
      // with the wedge, so the line behind strings out across the run.
      o.amp =
        o.role === 'teach'
          ? 1
          : clamp(
              o.amp + dt * 1.2 * ((o.v - target) / target),
              0.3,
              o.mode === 'plough' ? 0.7 : 1.75,
            )
      // Heading for the side of the run, they tighten the turn as the room runs out, down to half
      // the sidecut's radius with the ski right up on its edge, and start back before they need to.
      // A wedge turns tighter than any sidecut, and slowly.
      const sidecut = o.mode === 'plough' ? 6 : o.board ? CARVE.board : CARVE.ski
      const out = Math.sign(o.w) === Math.sign(Math.sin(o.psi))
      const room = Math.max(0.1, half - 1 - Math.abs(o.w))
      const need = out ? 1 - Math.cos(o.psi) : 0
      // Well over their speed, they check it: the skis thrown across the hill in a hard skid, a turn
      // that needs no more room than their own length.
      const check = o.mode !== 'plough' && o.v > target * 1.2
      const radius = check ? 2 : need > 0 ? clamp(room / need, sidecut / 2, sidecut) : sidecut
      const rate =
        o.mode === 'plough'
          ? o.role === 'teach'
            ? 0.8
            : 0.5
          : check
            ? 2.5
            : Math.max(o.v / radius, o.mode === 'skid' ? 1.4 : 0.6)
      o.psi += o.dir * rate * dt
      // What the edge holds across a carved turn. Asked for more, the ski lets go and skids, and the
      // further past it, the likelier it catches or washes out. A skidder is sliding already.
      if (o.mode === 'carve' && !check) over = (o.v * rate) / (G * (0.45 + o.skill))
      const edge = out && room < (check ? 1 : sidecut * 0.7) * need + 1
      // The instructor holds each traverse out to the side of the run before turning.
      if (o.role === 'teach' && o.dir * o.psi > o.amp) o.psi = o.dir * o.amp
      if ((o.dir * o.psi >= o.amp && o.role !== 'teach') || edge)
        o.dir = edge ? -Math.sign(o.w) : -o.dir
      if (turnAway && o.dir !== turnAway && o.dir * o.psi > 0.2) o.dir = turnAway
      // A teacher waits for the line to catch up.
      if (o.role === 'teach')
        for (const p of h.skiers) if (p.lead === o && p.state !== 'ski') want = 0
      if (o.mode !== 'plough') o.skid = over > 1 || check ? 1 : clamp((1 - o.skill) * 0.9, 0, 1)
    }
    if (o.mode === 'plough') {
      o.skid = clamp(o.skid + dt * (o.v - want), 0.1, 1)
      brake = 0.45 * G * Math.cos(a) * o.skid
    } else brake = 0.35 * G * o.skid * Math.abs(Math.sin(o.psi))
    const acc = down * Math.cos(o.psi) - glide - DRAG * o.v * o.v - brake
    o.v = Math.max(0, o.v + acc * dt)
    o.s += o.v * Math.cos(o.psi) * dt
    o.w += o.v * Math.sin(o.psi) * dt
  }
  if (Math.abs(o.w) > half) {
    o.w = Math.sign(o.w) * half
    o.v *= 0.9
  }
  o.s = Math.max(0, o.s)
  // Catching an edge: boarders more than skiers, beginners far more, more the faster they go, and a
  // beginner on a board's downhill edge most of all.
  const catchRate =
    (o.board ? 0.008 : 0.004) * (1 - o.skill) ** 2 * Math.min(1.5, o.v / 5) +
    (o.mode === 'leaf' ? 0.003 * (1 - o.conf) : 0) +
    Math.max(0, over - 1.3) * 0.1 * (1 - o.skill)
  if (h.rng() < catchRate * dt) return fall(h, o, voices)
  mark(h, o, run)
  if (o.role === 'teach' && o.trail.length) {
    const n = o.trail.length / 2
    const at = Math.floor(h.time * 4) % n
    o.trail[2 * at] = o.s
    o.trail[2 * at + 1] = o.w
  }
  if (o.mode === 'race' && o.run === 0)
    GATES.forEach((g, n) => {
      if (Math.abs(o.s - g.s) < o.v * dt + 0.2 && Math.abs(o.w - g.w) < 1.6) h.flick[n] = 0.4
    })
  if (o.s >= run.length - 6) {
    o.state = 'out'
    o.t = 0
  }
}

// Getting down and back up: the body slides to a stop, lies there a moment, sits up (a boarder
// rolls over onto their knees first), and goes on across the hill; with the skis up the slope they
// sidestep back up to them first and click in.
export function tumble(o: Skier, dt: number) {
  const run = RUNS[o.run]
  o.k = seek(run, o.s, o.k)
  if (o.state === 'down') {
    const a = run.pitch[o.k]
    o.v = Math.max(0, o.v - Math.max(1.5, SLIDE * G * Math.cos(a) - G * Math.sin(a)) * dt)
    o.s = Math.min(run.length - 7, o.s + o.v * dt)
    if (o.t > 1.5 + 2.5 * Math.min(1, o.hurt)) {
      o.state = o.hurt > 1 ? 'hurt' : 'up'
      o.t = 0
      if (o.hurt > 1) o.v = 0
    }
  } else if (o.state === 'up' && o.t > (o.board ? 3 : 2)) {
    o.state = o.gear >= 0 ? 'collect' : 'ski'
    o.t = 0
    o.v = 0
    o.psi = o.dir * 1.2
  } else if (o.state === 'collect') {
    const step = 0.5 * dt
    o.s = Math.max(o.gear, o.s - step)
    o.w += clamp(o.gearW - o.w, -step, step)
    if (o.s <= o.gear + 0.1 && o.t > 2) {
      o.gear = -1
      o.state = 'ski'
      o.v = 0
      o.psi = o.dir * 1.2
    }
  }
}
