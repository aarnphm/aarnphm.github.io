import { MathText } from '../../util/math-text'
import style from '../styles/jobDeadline.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const W = 640
const H = 220
const Y = 118
const R0 = 80
const R1 = 140
const DONE = 360
const DEAD = 500

let instance = 0

const JobDeadlineImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const id = ++instance

  return (
    <figure class="job-deadline">
      <svg
        class="jd-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Job timeline showing release window, response time, relative deadline, and absolute deadline."
      >
        <defs>
          <marker
            id={`jd-arrow-${id}`}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path class="jd-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>
        <line class="jd-axis" x1={40} y1={Y} x2={600} y2={Y} marker-end={`url(#jd-arrow-${id})`} />
        <text class="jd-axis-label" x={610} y={Y + 4}>
          Time
        </text>

        <rect class="jd-release" x={R0} y={Y - 18} width={R1 - R0} height={18} />
        <text class="jd-mark" x={R0} y={Y + 28} text-anchor="middle">
          rᵢ⁻
        </text>
        <text class="jd-mark" x={R1} y={Y + 28} text-anchor="middle">
          rᵢ⁺
        </text>

        <rect class="jd-job" x={R1} y={Y - 36} width={DONE - R1} height={28} />
        <text class="jd-job-label" x={(R1 + DONE) / 2} y={Y - 16} text-anchor="middle">
          Job Jᵢ
        </text>

        <line class="jd-tick" x1={DONE} y1={Y - 44} x2={DONE} y2={Y + 8} />
        <line class="jd-tick jd-tick--hard" x1={DEAD} y1={Y - 70} x2={DEAD} y2={Y + 8} />

        <line class="jd-span" x1={R1} y1={Y - 52} x2={DONE} y2={Y - 52} />
        <text class="jd-span-label" x={(R1 + DONE) / 2} y={Y - 58} text-anchor="middle">
          Response time
        </text>

        <line class="jd-span jd-span--rel" x1={R1} y1={Y - 78} x2={DEAD} y2={Y - 78} />
        <text class="jd-span-label" x={(R1 + DEAD) / 2} y={Y - 84} text-anchor="middle">
          Relative deadline Dᵢ
        </text>

        <text class="jd-anno" x={R1} y={Y + 48} text-anchor="middle">
          Release time rᵢ
        </text>
        <text class="jd-anno" x={DONE} y={Y + 48} text-anchor="middle">
          Completion time
        </text>
        <text class="jd-anno" x={DEAD} y={Y + 48} text-anchor="middle">
          Absolute deadline dᵢ
        </text>
      </svg>
      {caption ? (
        <figcaption class="jd-caption">
          <MathText text={caption} mathClass="jd-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const JobDeadlineComponent = JobDeadlineImpl as QuartzMdxComponent<Props>
JobDeadlineComponent.css = style

export const JobDeadline = registerMdxComponent('JobDeadline', JobDeadlineComponent)

export default (() => JobDeadline) satisfies (opts: undefined) => QuartzMdxComponent<Props>
