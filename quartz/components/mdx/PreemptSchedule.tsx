import { MathText } from '../../util/math-text'
import style from '../styles/preemptSchedule.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type Seg = { task: 1 | 2 | 3; x: number; w: number; done?: boolean }

const SEGS: Seg[] = [
  { task: 1, x: 40, w: 70 },
  { task: 2, x: 110, w: 55 },
  { task: 3, x: 165, w: 70, done: true },
  { task: 2, x: 235, w: 55, done: true },
  { task: 1, x: 290, w: 90 },
]

const Y: Record<1 | 2 | 3, number> = { 1: 150, 2: 95, 3: 40 }
const LABEL: Record<1 | 2 | 3, string> = { 1: 'Task1', 2: 'Task2', 3: 'Task3' }

const PreemptScheduleImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  return (
    <figure class="preempt-schedule">
      <svg
        class="ps-svg"
        viewBox="0 0 420 210"
        role="img"
        aria-label="Priority-based preemptive scheduling: Task3 preempts Task2, which preempts Task1."
      >
        <line class="ps-axis" x1={30} y1={180} x2={390} y2={180} />
        <line class="ps-axis" x1={30} y1={180} x2={30} y2={20} />
        <polygon class="ps-arrow" points="390,180 382,176 382,184" />
        <polygon class="ps-arrow" points="30,20 26,28 34,28" />
        <text class="ps-axis-label" x={210} y={200} text-anchor="middle">
          Time
        </text>
        <text class="ps-axis-label" x={14} y={100} text-anchor="middle" transform="rotate(-90 14 100)">
          Task Priority
        </text>
        <text class="ps-tick" x={36} y={154}>
          LOW
        </text>
        <text class="ps-tick" x={36} y={44}>
          HIGH
        </text>

        <line class="ps-xfer" x1={110} y1={150} x2={110} y2={95} />
        <line class="ps-xfer" x1={165} y1={95} x2={165} y2={40} />
        <line class="ps-xfer" x1={235} y1={40} x2={235} y2={95} />
        <line class="ps-xfer" x1={290} y1={95} x2={290} y2={150} />

        {SEGS.map((s, i) => (
          <g key={`seg-${i}`}>
            <rect
              class={s.task === 3 ? 'ps-bar ps-bar--lit' : 'ps-bar'}
              x={s.x}
              y={Y[s.task]}
              width={s.w}
              height={28}
            />
            <text class="ps-bar-label" x={s.x + s.w / 2} y={Y[s.task] + 18} text-anchor="middle">
              {LABEL[s.task]}
            </text>
            {s.done ? (
              <line
                class="ps-done"
                x1={s.x + s.w}
                y1={Y[s.task]}
                x2={s.x + s.w}
                y2={Y[s.task] + 28}
              />
            ) : null}
          </g>
        ))}

        <text class="ps-note" x={120} y={28}>
          Preemption
        </text>
        <text class="ps-note" x={300} y={28}>
          Task Completion
        </text>
      </svg>
      {caption ? (
        <figcaption class="ps-caption">
          <MathText text={caption} mathClass="ps-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const PreemptScheduleComponent = PreemptScheduleImpl as QuartzMdxComponent<Props>
PreemptScheduleComponent.css = style

export const PreemptSchedule = registerMdxComponent('PreemptSchedule', PreemptScheduleComponent)

export default (() => PreemptSchedule) satisfies (opts: undefined) => QuartzMdxComponent<Props>
