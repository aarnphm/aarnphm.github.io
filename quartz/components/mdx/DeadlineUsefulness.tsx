import { MathText } from '../../util/math-text'
import style from '../styles/deadlineUsefulness.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type Kind = 'soft' | 'firm' | 'hard'

const PANELS: { kind: Kind; title: string; path: string }[] = [
  { kind: 'soft', title: 'soft real time', path: 'M 20 30 H 90 L 150 110' },
  { kind: 'firm', title: 'firm real time', path: 'M 20 30 H 90 V 110' },
  { kind: 'hard', title: 'hard real time', path: 'M 20 30 H 90 V 150 H 150' },
]

const Panel = ({ kind, title, path }: { kind: Kind; title: string; path: string }) => (
  <div class="du-panel" data-du-kind={kind}>
    <p class="du-title">{title}</p>
    <svg
      class="du-svg"
      viewBox="0 0 170 170"
      role="img"
      aria-label={`${title} usefulness versus time`}
    >
      <line class="du-axis" x1={20} y1={110} x2={155} y2={110} />
      <line class="du-axis" x1={20} y1={110} x2={20} y2={15} />
      <polygon class="du-arrow" points="155,110 147,106 147,114" />
      <polygon class="du-arrow" points="20,15 16,23 24,23" />
      <text class="du-label" x={12} y={70} text-anchor="middle" transform="rotate(-90 12 70)">
        usefulness
      </text>
      <text class="du-label" x={90} y={128} text-anchor="middle">
        time
      </text>
      <line class="du-deadline" x1={90} y1={110} x2={90} y2={100} />
      <text class="du-deadline-label" x={90} y={142} text-anchor="middle">
        deadline
      </text>
      <path class={kind === 'soft' ? 'du-curve du-curve--lit' : 'du-curve'} d={path} />
    </svg>
  </div>
)

const DeadlineUsefulnessImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  return (
    <figure class="deadline-usefulness">
      <div class="du-row">
        {PANELS.map(p => (
          <Panel key={p.kind} {...p} />
        ))}
      </div>
      {caption ? (
        <figcaption class="du-caption">
          <MathText text={caption} mathClass="du-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const DeadlineUsefulnessComponent = DeadlineUsefulnessImpl as QuartzMdxComponent<Props>
DeadlineUsefulnessComponent.css = style

export const DeadlineUsefulness = registerMdxComponent(
  'DeadlineUsefulness',
  DeadlineUsefulnessComponent,
)

export default (() => DeadlineUsefulness) satisfies (opts: undefined) => QuartzMdxComponent<Props>
