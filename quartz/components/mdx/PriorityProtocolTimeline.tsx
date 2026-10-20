import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/priority-protocol-timeline.inline'
import style from '../styles/priorityProtocolTimeline.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Protocol = 'none' | 'pip' | 'pcp'
type PresetId = 'pip' | 'pcp' | 'chained'
type Props = { caption?: string; preset?: PresetId; protocol?: Protocol }

type Op = { run: number } | { lock: string } | { unlock: string }
type Task = {
  id: string
  name: string
  sub?: string
  tex: string
  base: number
  release: number
  ops: Op[]
}
type Preset = {
  id: PresetId
  label: string
  note: string
  horizon: number
  prio: (p: number) => string
  tasks: Task[]
}

const run = (n: number): Op => ({ run: n })
const lock = (r: string): Op => ({ lock: r })
const unlock = (r: string): Op => ({ unlock: r })
const HML = (p: number) => ['L', 'M', 'H'][p - 1]

// Larger base values mean higher priority, as in the note. The PIP slide labels its tasks by
// priority with 1 highest, so its labels print 5 - base.
const PRESETS: readonly Preset[] = [
  {
    id: 'pip',
    label: 'PIP slide',
    note: 'Slide task set: $T_1$ (priority 1, highest) to $T_4$ (priority 4). $T_1$ and $T_4$ share one resource $S$. Release times and segments read from the PIP slide; the slide does not show when $T_2$ and $T_3$ arrive, so they are assumed released at $t=9$ and $t=10$, inside $T_1$’s wait.',
    horizon: 28,
    prio: p => String(5 - p),
    tasks: [
      {
        id: 'T1',
        name: 'T',
        sub: '1',
        tex: 'T_1',
        base: 4,
        release: 5,
        ops: [run(3), lock('S'), run(3), unlock('S'), run(3)],
      },
      { id: 'T2', name: 'T', sub: '2', tex: 'T_2', base: 3, release: 9, ops: [run(3)] },
      { id: 'T3', name: 'T', sub: '3', tex: 'T_3', base: 2, release: 10, ops: [run(3)] },
      {
        id: 'T4',
        name: 'T',
        sub: '4',
        tex: 'T_4',
        base: 1,
        release: 0,
        ops: [run(3), lock('S'), run(6), unlock('S'), run(3)],
      },
    ],
  },
  {
    id: 'pcp',
    label: 'PCP slide',
    note: 'Slide task set: $C$ (H) locks $S_1$; $B$ (M) nests $S_3$ inside $S_2$; $A$ (L) nests $S_2$ inside $S_3$. Ceilings $S_1=\\mathrm{H}$, $S_2=S_3=\\mathrm{M}$. The slide has no time axis; durations are read from its bar lengths in whole units.',
    horizon: 36,
    prio: HML,
    tasks: [
      {
        id: 'C',
        name: 'C',
        tex: 'C',
        base: 3,
        release: 9,
        ops: [run(2), lock('S1'), run(4), unlock('S1'), run(2)],
      },
      {
        id: 'B',
        name: 'B',
        tex: 'B',
        base: 2,
        release: 4,
        ops: [run(2), lock('S2'), run(3), lock('S3'), run(3), unlock('S3'), run(3), unlock('S2')],
      },
      {
        id: 'A',
        name: 'A',
        tex: 'A',
        base: 1,
        release: 0,
        ops: [run(1), lock('S3'), run(6), lock('S2'), run(4), unlock('S2'), run(4), unlock('S3')],
      },
    ],
  },
  {
    id: 'chained',
    label: 'chained (synthetic)',
    note: 'Synthetic task set shaped after the chained-blocking case in Sha, Rajkumar and Lehoczky (1990): $T_H$ locks $S_1$ and then $S_2$; $T_M$ holds $S_2$; $T_L$ holds $S_1$. Both ceilings are H.',
    horizon: 20,
    prio: HML,
    tasks: [
      {
        id: 'TH',
        name: 'T',
        sub: 'H',
        tex: 'T_H',
        base: 3,
        release: 4,
        ops: [
          run(1),
          lock('S1'),
          run(2),
          unlock('S1'),
          run(1),
          lock('S2'),
          run(2),
          unlock('S2'),
          run(1),
        ],
      },
      {
        id: 'TM',
        name: 'T',
        sub: 'M',
        tex: 'T_M',
        base: 2,
        release: 2,
        ops: [run(1), lock('S2'), run(4), unlock('S2'), run(1)],
      },
      {
        id: 'TL',
        name: 'T',
        sub: 'L',
        tex: 'T_L',
        base: 1,
        release: 0,
        ops: [run(1), lock('S1'), run(4), unlock('S1'), run(1)],
      },
    ],
  },
]

const PROTOCOLS: readonly { id: Protocol; label: string; rule: string }[] = [
  {
    id: 'none',
    label: 'none',
    rule: 'No protocol. A task waiting for a held resource keeps its own priority, and the owner runs at its base priority, so medium-priority work can preempt the owner.',
  },
  {
    id: 'pip',
    label: 'PIP',
    rule: 'Priority inheritance. The owner of a resource inherits the effective priority of the tasks blocked on it, transitively. On unlock it recomputes from the waiters that remain.',
  },
  {
    id: 'pcp',
    label: 'PCP',
    rule: 'Priority ceiling. A task may lock a free resource only when $p_{\\mathrm{eff}}(T) > C_{-T}$, the highest ceiling among resources held by other tasks. The owner of the blocking resource inherits $T$’s priority.',
  },
]

type Step = {
  running: string | null
  eff: number
  holding: boolean
  pending: Record<string, string>
}
type Mark = { t: number; task: string; kind: 'lock' | 'unlock' | 'wait'; res: string }
type Request = { t: number; task: string; res: string; outcome: string }
type TaskResult = { blocking: number | null; sections: string[]; finish: number | null }
type Result = {
  steps: Step[]
  marks: Mark[]
  requests: Request[]
  tasks: Record<string, TaskResult>
  deadlock: { t: number; tasks: string[] } | null
  chained: string[]
  end: number
}

const ceilings = (p: Preset): Record<string, number> => {
  const out: Record<string, number> = {}
  for (const t of p.tasks)
    for (const op of t.ops) if ('lock' in op) out[op.lock] = Math.max(out[op.lock] ?? 0, t.base)
  return out
}

const simulate = (preset: Preset, protocol: Protocol): Result => {
  const tasks = preset.tasks
  const ceil = ceilings(preset)
  const byId = Object.fromEntries(tasks.map(t => [t.id, t]))
  const label = (id: string) => byId[id].tex
  const prioTex = (p: number) => `\\mathrm{${preset.prio(p)}}`
  const pc: Record<string, number> = {}
  const left: Record<string, number> = {}
  const held: Record<string, string[]> = {}
  const pending: Record<string, string | null> = {}
  const done: Record<string, number | null> = {}
  for (const t of tasks) {
    pc[t.id] = 0
    left[t.id] = 'run' in t.ops[0] ? t.ops[0].run : 0
    held[t.id] = []
    pending[t.id] = null
    done[t.id] = null
  }
  const owner = (res: string) => tasks.find(t => held[t.id].includes(res))?.id ?? null
  const active = (id: string, t: number) => byId[id].release <= t && done[id] === null

  // Why a lock attempt fails, or null when it succeeds.
  type Block = { by: string; why: 'held' | 'ceiling'; res: string }
  const check = (id: string, res: string, eff: Record<string, number>): Block | null => {
    const holder = owner(res)
    if (holder && holder !== id) return { by: holder, why: 'held', res }
    if (protocol !== 'pcp') return null
    const others = tasks
      .filter(t => t.id !== id)
      .flatMap(t => held[t.id])
      .sort((a, b) => ceil[b] - ceil[a])
    if (others.length && !(eff[id] > ceil[others[0]]))
      return { by: owner(others[0])!, why: 'ceiling', res: others[0] }
    return null
  }

  const effective = (): Record<string, number> => {
    let eff = Object.fromEntries(tasks.map(t => [t.id, t.base]))
    if (protocol === 'none') return eff
    for (let iter = 0; iter < 16; iter++) {
      const next = Object.fromEntries(tasks.map(t => [t.id, t.base]))
      for (let k = 0; k < tasks.length; k++)
        for (const t of tasks) {
          const res = pending[t.id]
          if (!res || done[t.id] !== null) continue
          const b = check(t.id, res, next)
          if (b) next[b.by] = Math.max(next[b.by], next[t.id])
        }
      const same = tasks.every(t => next[t.id] === eff[t.id])
      eff = next
      if (same) break
    }
    return eff
  }

  const finishUnlocks = (id: string, t: number) => {
    const ops = byId[id].ops
    while (pc[id] < ops.length) {
      const op = ops[pc[id]]
      if ('unlock' in op) {
        held[id] = held[id].filter(r => r !== op.unlock)
        marks.push({ t, task: id, kind: 'unlock', res: op.unlock })
        pc[id]++
      } else {
        if ('run' in op) left[id] = op.run
        break
      }
    }
    if (pc[id] >= ops.length) done[id] = t
  }

  const steps: Step[] = []
  const marks: Mark[] = []
  const requests: Request[] = []
  const blocking: Record<string, number> = Object.fromEntries(tasks.map(t => [t.id, 0]))
  const sections: Record<string, string[]> = Object.fromEntries(tasks.map(t => [t.id, []]))
  let deadlock: Result['deadlock'] = null
  let last: string | null = null
  let end = preset.horizon

  for (let t = 0; t < preset.horizon && !deadlock; t++) {
    let chosen: string | null = null
    let eff = effective()
    for (let guard = 0; guard < 64; guard++) {
      eff = effective()
      const runnable = tasks.filter(
        k => active(k.id, t) && (!pending[k.id] || !check(k.id, pending[k.id]!, eff)),
      )
      if (!runnable.length) break
      const pick = runnable.reduce((a, b) =>
        eff[b.id] > eff[a.id] || (eff[b.id] === eff[a.id] && b.id === last) ? b : a,
      )
      const op = pick.ops[pc[pick.id]]
      if ('lock' in op) {
        const block = check(pick.id, op.lock, eff)
        if (block) {
          pending[pick.id] = op.lock
          marks.push({ t, task: pick.id, kind: 'wait', res: op.lock })
          const outcome =
            block.why === 'held'
              ? `blocked: $${block.res.replace(/(\d)/, '_$1')}$ held by $${label(block.by)}$`
              : `blocked: $p_{\\mathrm{eff}}=${prioTex(eff[pick.id])}$ not above $C_{-${label(pick.id)}}=${prioTex(ceil[block.res])}$ ($${block.res.replace(/(\d)/, '_$1')}$ held by $${label(block.by)}$)`
          requests.push({ t, task: pick.id, res: op.lock, outcome })
          continue
        }
        const others = tasks.filter(k => k.id !== pick.id).flatMap(k => held[k.id])
        const top = others.sort((a, b) => ceil[b] - ceil[a])[0]
        const outcome =
          protocol === 'pcp' && top
            ? `granted: $p_{\\mathrm{eff}}=${prioTex(eff[pick.id])}$ above $C_{-${label(pick.id)}}=${prioTex(ceil[top])}$`
            : pending[pick.id]
              ? 'granted after waiting'
              : 'granted'
        requests.push({ t, task: pick.id, res: op.lock, outcome })
        pending[pick.id] = null
        held[pick.id].push(op.lock)
        marks.push({ t, task: pick.id, kind: 'lock', res: op.lock })
        pc[pick.id]++
        const nextOp = pick.ops[pc[pick.id]]
        if (nextOp && 'run' in nextOp) left[pick.id] = nextOp.run
        continue
      }
      chosen = pick.id
      break
    }

    if (chosen === null) {
      const waiting = tasks.filter(k => active(k.id, t) && pending[k.id])
      if (waiting.length) {
        // Nothing can run and some task waits: follow the blockers until one repeats.
        let cur = waiting[0].id
        const seen: string[] = []
        while (!seen.includes(cur) && pending[cur]) {
          seen.push(cur)
          cur = check(cur, pending[cur]!, eff)!.by
        }
        deadlock = { t, tasks: seen.slice(seen.indexOf(cur)) }
        end = t
        break
      }
    }

    steps.push({
      running: chosen,
      eff: chosen ? eff[chosen] : 0,
      holding: chosen ? held[chosen].length > 0 : false,
      pending: Object.fromEntries(
        tasks.filter(k => active(k.id, t) && pending[k.id]).map(k => [k.id, pending[k.id]!]),
      ),
    })
    for (const k of tasks) {
      if (!active(k.id, t) || k.id === chosen || !chosen) continue
      if (byId[chosen].base >= k.base) continue
      blocking[k.id]++
      const key = held[chosen].length ? `${chosen}:${held[chosen][0]}` : chosen
      if (!sections[k.id].includes(key)) sections[k.id].push(key)
    }
    if (chosen) {
      last = chosen
      left[chosen]--
      if (left[chosen] === 0) {
        pc[chosen]++
        finishUnlocks(chosen, t + 1)
      }
    }
  }

  const dead = new Set(deadlock?.tasks ?? [])
  const taskResults = Object.fromEntries(
    tasks.map(k => [
      k.id,
      {
        blocking: dead.has(k.id) ? null : blocking[k.id],
        sections: sections[k.id],
        finish: done[k.id],
      },
    ]),
  )
  const chained = tasks
    .filter(k => sections[k.id].filter(s => s.includes(':')).length >= 2)
    .map(k => k.id)
  return { steps, marks, requests, tasks: taskResults, deadlock, chained, end }
}

const STATES: Record<string, Result> = {}
for (const p of PRESETS) for (const pr of PROTOCOLS) STATES[`${p.id}:${pr.id}`] = simulate(p, pr.id)

// Gantt geometry.
const VIEW_W = 600
const LABEL_W = 34
const PLOT_R = VIEW_W - 8
const TOP = 20
const ROW_H = 36
const BAR_H = 15
const BAR_Y = 13
const WAIT_H = 5

type Segment = {
  task: string
  t0: number
  t1: number
  eff: number
  kind: 'run' | 'held' | 'inherit'
}

const segments = (p: Preset, r: Result): Segment[] => {
  const out: Segment[] = []
  r.steps.forEach((s, t) => {
    if (!s.running) return
    const base = p.tasks.find(k => k.id === s.running)!.base
    const kind = s.eff > base ? 'inherit' : s.holding ? 'held' : 'run'
    const prev = out[out.length - 1]
    // A lock or unlock splits a bar, so its rule falls on a segment edge, not through a label.
    const split = r.marks.some(m => m.task === s.running && m.t === t && m.kind !== 'wait')
    if (
      prev &&
      !split &&
      prev.task === s.running &&
      prev.t1 === t &&
      prev.eff === s.eff &&
      prev.kind === kind
    )
      prev.t1 = t + 1
    else out.push({ task: s.running, t0: t, t1: t + 1, eff: s.eff, kind })
  })
  return out
}

const waits = (p: Preset, r: Result) =>
  p.tasks.flatMap(k => {
    const out: { task: string; t0: number; t1: number }[] = []
    const n = r.deadlock ? p.horizon : r.steps.length
    for (let t = 0; t < n; t++) {
      const on =
        t < r.steps.length ? Boolean(r.steps[t].pending[k.id]) : r.deadlock!.tasks.includes(k.id)
      const prev = out[out.length - 1]
      if (!on) continue
      if (prev && prev.t1 === t) prev.t1 = t + 1
      else out.push({ task: k.id, t0: t, t1: t + 1 })
    }
    return out
  })

const resText = (r: string) => r.replace(/(\d)/, (d: string) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)])

const Gantt = ({ p, r, label }: { p: Preset; r: Result; label: string }) => {
  const unit = (PLOT_R - LABEL_W) / p.horizon
  const x = (t: number) => LABEL_W + t * unit
  const rowY = (i: number) => TOP + i * ROW_H
  const axisY = rowY(p.tasks.length) + 2
  const viewH = axisY + 20
  const rowOf = (id: string) => p.tasks.findIndex(k => k.id === id)
  const step = p.horizon > 30 ? 5 : 2
  return (
    <svg class="ppt-svg" viewBox={`0 0 ${VIEW_W} ${viewH}`} role="img" aria-label={label}>
      {p.tasks.map((k, i) => (
        <g class="ppt-row">
          <text class="ppt-task" x={4} y={rowY(i) + BAR_Y + 11}>
            {k.name}
            {k.sub ? (
              <tspan class="ppt-task-sub" dy={3}>
                {k.sub}
              </tspan>
            ) : null}
          </text>
          <line
            class="ppt-baseline"
            x1={LABEL_W}
            y1={rowY(i) + BAR_Y + BAR_H}
            x2={PLOT_R}
            y2={rowY(i) + BAR_Y + BAR_H}
          />
          <polygon
            class="ppt-release"
            points={`${x(k.release)},${rowY(i) + BAR_Y + BAR_H} ${x(k.release) - 3.5},${rowY(i) + BAR_Y + BAR_H + 6} ${x(k.release) + 3.5},${rowY(i) + BAR_Y + BAR_H + 6}`}
          />
        </g>
      ))}
      {waits(p, r).map(w => {
        const i = rowOf(w.task)
        return (
          <rect
            class="ppt-wait"
            data-ppt-wait={`${w.task}:${w.t0}:${w.t1}`}
            x={x(w.t0)}
            y={rowY(i) + BAR_Y + (BAR_H - WAIT_H) / 2}
            width={x(w.t1) - x(w.t0)}
            height={WAIT_H}
          />
        )
      })}
      {segments(p, r).map(s => {
        const i = rowOf(s.task)
        const w = x(s.t1) - x(s.t0)
        return (
          <g>
            <rect
              class={`ppt-seg ppt-seg--${s.kind}`}
              data-ppt-seg={`${s.task}:${s.t0}:${s.t1}:${s.kind}`}
              x={x(s.t0)}
              y={rowY(i) + BAR_Y}
              width={w}
              height={BAR_H}
            />
            {w >= 9 ? (
              <text
                class="ppt-prio"
                x={x(s.t0) + w / 2}
                y={rowY(i) + BAR_Y + 11}
                text-anchor="middle"
              >
                {p.prio(s.eff)}
              </text>
            ) : null}
          </g>
        )
      })}
      {r.marks.map(m => {
        const i = rowOf(m.task)
        const y = rowY(i) + BAR_Y
        const sign = m.kind === 'lock' ? '+' : m.kind === 'unlock' ? '−' : '?'
        return (
          <g class={`ppt-mark ppt-mark--${m.kind}`}>
            {m.kind === 'wait' ? null : <line x1={x(m.t)} y1={y} x2={x(m.t)} y2={y + BAR_H} />}
            <text
              x={m.kind === 'wait' ? x(m.t) + 3 : x(m.t)}
              y={y - 3}
              text-anchor={m.kind === 'wait' ? 'start' : 'middle'}
            >
              {`${sign}${resText(m.res)}`}
            </text>
          </g>
        )
      })}
      {r.deadlock ? (
        <g class="ppt-deadlock">
          <line x1={x(r.deadlock.t)} y1={TOP - 2} x2={x(r.deadlock.t)} y2={axisY} />
          <text x={x(r.deadlock.t) + 4} y={TOP - 6}>
            {`deadlock at t = ${r.deadlock.t}`}
          </text>
        </g>
      ) : null}
      <line class="ppt-axis" x1={LABEL_W} y1={axisY} x2={PLOT_R} y2={axisY} />
      {Array.from({ length: p.horizon + 1 }, (_, t) => (
        <g>
          <line
            class="ppt-tick"
            x1={x(t)}
            y1={axisY}
            x2={x(t)}
            y2={axisY + (t % step === 0 ? 4 : 2)}
          />
          {t % step === 0 ? (
            <text class="ppt-axis-label" x={x(t)} y={axisY + 13} text-anchor="middle">
              {t}
            </text>
          ) : null}
        </g>
      ))}
    </svg>
  )
}

const sectionText = (p: Preset, key: string) => {
  const [task, res] = key.split(':')
  const t = p.tasks.find(k => k.id === task)!.tex
  return res ? `$${t}$ in $${res.replace(/(\d)/, '_$1')}$` : `$${t}$`
}

const ganttLabel = (p: Preset, r: Result, protocol: string) =>
  `${p.label} under ${protocol}: ` +
  p.tasks
    .map(k => {
      const tr = r.tasks[k.id]
      return `${k.id} ${tr.blocking === null ? 'deadlocked' : `blocked ${tr.blocking} units, done at ${tr.finish}`}`
    })
    .join('; ')

let instance = 0

const PriorityProtocolTimelineImpl: QuartzMdxComponent<Props> = ({
  caption,
  preset = 'pip',
  protocol = 'pip',
}) => {
  const id = `ppt-${++instance}`
  const p0 = PRESETS.some(p => p.id === preset) ? preset : 'pip'
  const pr0 = PROTOCOLS.some(p => p.id === protocol) ? protocol : 'pip'
  const key0 = `${p0}:${pr0}`

  return (
    <figure class="priority-protocol-timeline" data-priority-protocol-timeline data-ppt-key={key0}>
      <div class="ppt-controls">
        <fieldset class="ppt-segments">
          <legend class="ppt-legend">Task set</legend>
          {PRESETS.map(p => (
            <label class="ppt-segment">
              <input
                type="radio"
                name={`${id}-preset`}
                value={p.id}
                checked={p.id === p0}
                data-ppt-preset
              />
              <span>{p.label}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="ppt-segments">
          <legend class="ppt-legend">Protocol</legend>
          {PROTOCOLS.map(p => (
            <label class="ppt-segment">
              <input
                type="radio"
                name={`${id}-protocol`}
                value={p.id}
                checked={p.id === pr0}
                data-ppt-protocol
              />
              <span>{p.label}</span>
            </label>
          ))}
        </fieldset>
      </div>

      {PRESETS.map(p => (
        <p class="ppt-note" data-ppt-preset-panel={p.id} hidden={p.id !== p0}>
          <MathText text={p.note} mathClass="ppt-math" />
        </p>
      ))}
      {PROTOCOLS.map(p => (
        <p class="ppt-rule" data-ppt-protocol-panel={p.id} hidden={p.id !== pr0}>
          <MathText text={p.rule} mathClass="ppt-math" />
        </p>
      ))}

      {PRESETS.flatMap(p =>
        PROTOCOLS.map(pr => {
          const key = `${p.id}:${pr.id}`
          const r = STATES[key]
          return (
            <div class="ppt-panel" data-ppt-panel={key} hidden={key !== key0}>
              <div class="ppt-scroll">
                <Gantt p={p} r={r} label={ganttLabel(p, r, pr.label)} />
              </div>
              <ul class="ppt-key">
                <li>
                  <span class="ppt-swatch ppt-swatch--run" />
                  runs at base priority
                </li>
                <li>
                  <span class="ppt-swatch ppt-swatch--held" />
                  holds a resource
                </li>
                <li>
                  <span class="ppt-swatch ppt-swatch--inherit" />
                  runs at an inherited priority
                </li>
                <li>
                  <span class="ppt-swatch ppt-swatch--wait" />
                  waits for a lock
                </li>
              </ul>
              <table class="ppt-table">
                <thead>
                  <tr>
                    <th scope="col">task</th>
                    <th scope="col">base</th>
                    <th scope="col">release</th>
                    <th scope="col">blocking</th>
                    <th scope="col">blocked by</th>
                    <th scope="col">done</th>
                  </tr>
                </thead>
                <tbody>
                  {p.tasks.map(k => {
                    const tr = r.tasks[k.id]
                    return (
                      <tr data-ppt-task={k.id}>
                        <th scope="row">
                          <MathText text={`$${k.tex}$`} mathClass="ppt-math" />
                        </th>
                        <td>{p.prio(k.base)}</td>
                        <td>{k.release}</td>
                        <td data-ppt-blocking>{tr.blocking === null ? '∞' : tr.blocking}</td>
                        <td class="ppt-by">
                          {tr.sections.length ? (
                            <MathText
                              text={tr.sections.map(s => sectionText(p, s)).join(', ')}
                              mathClass="ppt-math"
                            />
                          ) : (
                            '–'
                          )}
                        </td>
                        <td data-ppt-finish>{tr.finish === null ? '–' : tr.finish}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              <dl class="ppt-readout">
                <div>
                  <dt>Chained blocking</dt>
                  <dd data-ppt-out="chained">
                    {r.chained.length ? (
                      <MathText
                        text={`yes: ${r.chained.map(c => `$${p.tasks.find(k => k.id === c)!.tex}$`).join(', ')} waits on ${Math.max(...r.chained.map(c => r.tasks[c].sections.filter(s => s.includes(':')).length))} critical sections`}
                        mathClass="ppt-math"
                      />
                    ) : (
                      'no'
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Deadlock</dt>
                  <dd data-ppt-out="deadlock">
                    {r.deadlock ? (
                      <MathText
                        text={`at $t=${r.deadlock.t}$, ${r.deadlock.tasks.map(d => `$${p.tasks.find(k => k.id === d)!.tex}$`).join(' and ')} wait on each other`}
                        mathClass="ppt-math"
                      />
                    ) : (
                      'no'
                    )}
                  </dd>
                </div>
              </dl>
              <table class="ppt-table ppt-requests">
                <thead>
                  <tr>
                    <th scope="col">t</th>
                    <th scope="col">task</th>
                    <th scope="col">lock</th>
                    <th scope="col">outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {r.requests.map(q => (
                    <tr>
                      <td>{q.t}</td>
                      <td>
                        <MathText
                          text={`$${p.tasks.find(k => k.id === q.task)!.tex}$`}
                          mathClass="ppt-math"
                        />
                      </td>
                      <td>
                        <MathText text={`$${q.res.replace(/(\d)/, '_$1')}$`} mathClass="ppt-math" />
                      </td>
                      <td class="ppt-outcome">
                        <MathText text={q.outcome} mathClass="ppt-math" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }),
      )}

      {caption ? (
        <figcaption class="ppt-caption">
          <MathText text={caption} mathClass="ppt-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const PriorityProtocolTimelineComponent = PriorityProtocolTimelineImpl as QuartzMdxComponent<Props>
PriorityProtocolTimelineComponent.css = style
PriorityProtocolTimelineComponent.afterDOMLoaded = script

export const PriorityProtocolTimeline = registerMdxComponent(
  'PriorityProtocolTimeline',
  PriorityProtocolTimelineComponent,
)

export default (() => PriorityProtocolTimeline) satisfies (
  opts: undefined,
) => QuartzMdxComponent<Props>
