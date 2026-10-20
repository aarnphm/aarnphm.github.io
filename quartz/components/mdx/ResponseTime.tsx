import { MathText } from '../../util/math-text'
import style from '../styles/responseTime.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const ARRIVAL = 80
const START = 220
const DONE = 420
const Y = 90

let instance = 0

const ResponseTimeImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const id = ++instance

  return (
    <figure class="response-time">
      <svg
        class="rt-svg"
        viewBox="0 0 560 200"
        role="img"
        aria-label="Timeline of wait time, execution time, and response time for a job."
      >
        <defs>
          <marker
            id={`rt-arrow-${id}`}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path class="rt-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <line class="rt-axis" x1={40} y1={Y} x2={520} y2={Y} marker-end={`url(#rt-arrow-${id})`} />
        <text class="rt-axis-label" x={530} y={Y + 4}>
          Time
        </text>

        {[
          { x: ARRIVAL, label: 'Arrival' },
          { x: START, label: 'Execution' },
          { x: DONE, label: 'output' },
        ].map(m => (
          <g key={m.label}>
            <line class="rt-marker" x1={m.x} y1={Y - 28} x2={m.x} y2={Y} />
            <text class="rt-marker-label" x={m.x} y={Y - 34} text-anchor="middle">
              {m.label}
            </text>
          </g>
        ))}

        <rect class="rt-job" x={START} y={Y + 10} width={DONE - START} height={28} />
        <text class="rt-job-label" x={(START + DONE) / 2} y={Y + 29} text-anchor="middle">
          Job run
        </text>

        <line class="rt-bracket" x1={ARRIVAL} y1={Y + 56} x2={START} y2={Y + 56} />
        <text class="rt-bracket-label" x={(ARRIVAL + START) / 2} y={Y + 72} text-anchor="middle">
          Wait Time
        </text>

        <line class="rt-bracket" x1={START} y1={Y + 56} x2={DONE} y2={Y + 56} />
        <text class="rt-bracket-label" x={(START + DONE) / 2} y={Y + 72} text-anchor="middle">
          Execution Time
        </text>

        <line class="rt-bracket rt-bracket--wide" x1={ARRIVAL} y1={Y + 92} x2={DONE} y2={Y + 92} />
        <text class="rt-bracket-label" x={(ARRIVAL + DONE) / 2} y={Y + 108} text-anchor="middle">
          Response Time
        </text>
      </svg>
      {caption ? (
        <figcaption class="rt-caption">
          <MathText text={caption} mathClass="rt-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ResponseTimeComponent = ResponseTimeImpl as QuartzMdxComponent<Props>
ResponseTimeComponent.css = style

export const ResponseTime = registerMdxComponent('ResponseTime', ResponseTimeComponent)

export default (() => ResponseTime) satisfies (opts: undefined) => QuartzMdxComponent<Props>
