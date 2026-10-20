import { MathText } from '../../util/math-text'
import style from '../styles/packetSniffer.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const W = 640
const H = 280
const APP_Y = 28
const OS_Y = 148
const DIV_Y = 132
const STACK = [
  { y: 156, label: 'Transport (TCP/UDP)' },
  { y: 186, label: 'Network (IP)' },
  { y: 216, label: 'Link (Ethernet)' },
  { y: 246, label: 'Physical' },
] as const

let instance = 0

const PacketSnifferImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `psn-arrow-${++instance}`

  return (
    <figure class="packet-sniffer">
      <svg
        class="psn-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Packet capture copies Ethernet frames from the OS to an application-layer packet analyzer."
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path class="psn-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <rect class="psn-frame" x={16} y={12} width={608} height={256} />
        <line class="psn-div" x1={16} y1={DIV_Y} x2={624} y2={DIV_Y} />
        <text class="psn-tier" x={28} y={APP_Y + 14}>
          application
        </text>
        <text class="psn-tier" x={28} y={OS_Y + 14}>
          operating system
        </text>

        <rect class="psn-box" x={400} y={48} width={200} height={56} />
        <text class="psn-label" x={500} y={80} text-anchor="middle">
          application
        </text>
        <text class="psn-sub" x={500} y={96} text-anchor="middle">
          browser, ftp, …
        </text>

        {STACK.map(s => (
          <g key={s.label}>
            <rect class="psn-box" x={420} y={s.y} width={160} height={24} />
            <text class="psn-label" x={500} y={s.y + 16} text-anchor="middle">
              {s.label}
            </text>
          </g>
        ))}

        <line class="psn-edge" x1={500} y1={156} x2={500} y2={104} marker-end={`url(#${arrow})`} />
        <path
          class="psn-edge"
          d="M440 276 H560"
          marker-start={`url(#${arrow})`}
          marker-end={`url(#${arrow})`}
        />
        <line class="psn-edge" x1={500} y1={270} x2={500} y2={276} />
        <text class="psn-sub" x={500} y={H - 4} text-anchor="middle">
          to/from network
        </text>

        <rect class="psn-sniff" x={120} y={40} width={200} height={220} />
        <text class="psn-tier" x={220} y={58} text-anchor="middle">
          packet sniffer
        </text>
        <rect class="psn-box psn-box--lit" x={140} y={72} width={160} height={40} />
        <text class="psn-label" x={220} y={96} text-anchor="middle">
          packet analyzer
        </text>
        <rect class="psn-box psn-box--lit" x={140} y={200} width={160} height={40} />
        <text class="psn-label" x={220} y={224} text-anchor="middle">
          packet capture
        </text>
        <text class="psn-sub" x={220} y={238} text-anchor="middle">
          pcap
        </text>
        <line
          class="psn-edge psn-edge--strong"
          x1={220}
          y1={200}
          x2={220}
          y2={112}
          marker-end={`url(#${arrow})`}
        />
        <line class="psn-edge psn-edge--strong" x1={220} y1={112} x2={220} y2={200} />
        <path class="psn-edge psn-edge--tap" d="M420 228 H300" marker-end={`url(#${arrow})`} />
        <text class="psn-tap" x={360} y={218} text-anchor="middle">
          copy of Ethernet frames
        </text>
      </svg>

      {caption ? (
        <figcaption class="psn-caption">
          <MathText text={caption} mathClass="psn-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const PacketSnifferComponent = PacketSnifferImpl as QuartzMdxComponent<Props>
PacketSnifferComponent.css = style

export const PacketSniffer = registerMdxComponent('PacketSniffer', PacketSnifferComponent)

export default (() => PacketSniffer) satisfies (opts: undefined) => QuartzMdxComponent<Props>
