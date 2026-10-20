import { MathText } from '../../util/math-text'
import style from '../styles/threadAddressSpace.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const SHARED = ['Global Data', 'Code', 'Heap'] as const
const W = 640
const H = 300

let instance = 0

const ThreadAddressSpaceImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `tas-arrow-${++instance}`
  const processX = 250
  const processW = 140
  const threadW = 120
  const leftX = 40
  const rightX = 480
  const rowH = 36
  const processRows = [...SHARED, 'Stack for Thread 1', 'Stack for Thread 2']
  const threadRows = [...SHARED, 'Stack']

  return (
    <figure class="thread-address-space">
      <svg
        class="tas-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Process address space shared by two threads, each with its own stack."
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path class="tas-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <rect class="tas-frame" x={processX} y={36} width={processW} height={processRows.length * rowH + 28} />
        <text class="tas-title" x={processX + processW / 2} y={28} text-anchor="middle">
          Process Address Space
        </text>
        {processRows.map((label, i) => {
          const y = 52 + i * rowH
          const lit = label.startsWith('Stack')
          return (
            <g key={label}>
              <rect
                class={lit ? 'tas-cell tas-cell--private' : 'tas-cell tas-cell--shared'}
                x={processX + 10}
                y={y}
                width={processW - 20}
                height={rowH - 8}
              />
              <text class="tas-label" x={processX + processW / 2} y={y + 18} text-anchor="middle">
                {label}
              </text>
            </g>
          )
        })}

        {[
          { x: leftX, title: "Thread 2's Address Space", stackTarget: 4 },
          { x: rightX, title: "Thread 1's Address Space", stackTarget: 3 },
        ].map(side => (
          <g key={side.title}>
            <rect
              class="tas-frame tas-frame--thread"
              x={side.x}
              y={52}
              width={threadW}
              height={threadRows.length * rowH + 20}
            />
            <text class="tas-title" x={side.x + threadW / 2} y={44} text-anchor="middle">
              {side.title}
            </text>
            {threadRows.map((label, i) => {
              const y = 68 + i * rowH
              const targetI = label === 'Stack' ? side.stackTarget : i
              const targetY = 52 + targetI * rowH + (rowH - 8) / 2
              const fromX = side.x < processX ? side.x + threadW : side.x
              const toX = side.x < processX ? processX : processX + processW
              return (
                <g key={`${side.title}-${label}`}>
                  <rect
                    class="tas-cell tas-cell--view"
                    x={side.x + 8}
                    y={y}
                    width={threadW - 16}
                    height={rowH - 8}
                  />
                  <text class="tas-label" x={side.x + threadW / 2} y={y + 18} text-anchor="middle">
                    {label}
                  </text>
                  <line
                    class={label === 'Stack' ? 'tas-edge tas-edge--private' : 'tas-edge'}
                    x1={fromX}
                    y1={y + (rowH - 8) / 2}
                    x2={toX}
                    y2={targetY}
                    marker-end={`url(#${arrow})`}
                  />
                </g>
              )
            })}
          </g>
        ))}
      </svg>
      {caption ? (
        <figcaption class="tas-caption">
          <MathText text={caption} mathClass="tas-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ThreadAddressSpaceComponent = ThreadAddressSpaceImpl as QuartzMdxComponent<Props>
ThreadAddressSpaceComponent.css = style

export const ThreadAddressSpace = registerMdxComponent(
  'ThreadAddressSpace',
  ThreadAddressSpaceComponent,
)

export default (() => ThreadAddressSpace) satisfies (opts: undefined) => QuartzMdxComponent<Props>
