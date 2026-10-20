import { MathText } from '../../util/math-text'
import style from '../styles/bPlusTree.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type Node = { keys: number[]; x: number; y: number; w: number }
type Edge = { x1: number; y1: number; x2: number; y2: number }

const CELL = 22
const NODE_H = 28
const PTR = 8

const nodeWidth = (keys: number[]) => keys.length * CELL + (keys.length + 1) * PTR + 4

const L1: Node = { keys: [30, 65], x: 300, y: 36, w: 0 }
L1.w = nodeWidth(L1.keys)

const L2: Node[] = [
  { keys: [15, 21], x: 110, y: 120, w: 0 },
  { keys: [41], x: 300, y: 120, w: 0 },
  { keys: [78], x: 470, y: 120, w: 0 },
]
for (const n of L2) n.w = nodeWidth(n.keys)

const L3: Node[] = [
  { keys: [2, 7], x: 36, y: 220, w: 0 },
  { keys: [16, 20], x: 120, y: 220, w: 0 },
  { keys: [22, 29], x: 204, y: 220, w: 0 },
  { keys: [35, 37], x: 288, y: 220, w: 0 },
  { keys: [42, 50], x: 372, y: 220, w: 0 },
  { keys: [67, 77], x: 456, y: 220, w: 0 },
  { keys: [80, 91], x: 540, y: 220, w: 0 },
]
for (const n of L3) n.w = nodeWidth(n.keys)

const ptrX = (n: Node, i: number) => n.x - n.w / 2 + 2 + PTR / 2 + i * (CELL + PTR)

const edges: Edge[] = [
  { x1: ptrX(L1, 0), y1: L1.y + NODE_H, x2: L2[0].x, y2: L2[0].y },
  { x1: ptrX(L1, 1), y1: L1.y + NODE_H, x2: L2[1].x, y2: L2[1].y },
  { x1: ptrX(L1, 2), y1: L1.y + NODE_H, x2: L2[2].x, y2: L2[2].y },
  { x1: ptrX(L2[0], 0), y1: L2[0].y + NODE_H, x2: L3[0].x, y2: L3[0].y },
  { x1: ptrX(L2[0], 1), y1: L2[0].y + NODE_H, x2: L3[1].x, y2: L3[1].y },
  { x1: ptrX(L2[0], 2), y1: L2[0].y + NODE_H, x2: L3[2].x, y2: L3[2].y },
  { x1: ptrX(L2[1], 0), y1: L2[1].y + NODE_H, x2: L3[3].x, y2: L3[3].y },
  { x1: ptrX(L2[1], 1), y1: L2[1].y + NODE_H, x2: L3[4].x, y2: L3[4].y },
  { x1: ptrX(L2[2], 0), y1: L2[2].y + NODE_H, x2: L3[5].x, y2: L3[5].y },
  { x1: ptrX(L2[2], 1), y1: L2[2].y + NODE_H, x2: L3[6].x, y2: L3[6].y },
]

let instance = 0

const drawNode = (n: Node, lit: boolean, key: string) => {
  const left = n.x - n.w / 2
  return (
    <g key={key}>
      <rect
        class={lit ? 'bpt-node bpt-node--lit' : 'bpt-node'}
        x={left}
        y={n.y}
        width={n.w}
        height={NODE_H}
      />
      {n.keys.map((k, i) => {
        const cx = left + 2 + PTR + i * (CELL + PTR) + CELL / 2
        return (
          <g key={`${key}-k-${i}`}>
            <circle class="bpt-ptr" cx={left + 2 + PTR / 2 + i * (CELL + PTR)} cy={n.y + NODE_H / 2} r={2.2} />
            <rect
              class="bpt-cell"
              x={left + 2 + PTR + i * (CELL + PTR)}
              y={n.y + (NODE_H - CELL + 4) / 2}
              width={CELL}
              height={CELL - 4}
            />
            <text class="bpt-key" x={cx} y={n.y + NODE_H / 2 + 4} text-anchor="middle">
              {k}
            </text>
          </g>
        )
      })}
      <circle
        class="bpt-ptr"
        cx={left + 2 + PTR / 2 + n.keys.length * (CELL + PTR)}
        cy={n.y + NODE_H / 2}
        r={2.2}
      />
    </g>
  )
}

const BPlusTreeImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `bpt-arrow-${++instance}`

  return (
    <figure class="b-plus-tree">
      <svg
        class="bpt-svg"
        viewBox="0 0 640 300"
        role="img"
        aria-label="B+ tree with root keys 30 and 65, three internal nodes, and seven leaf nodes holding data entries."
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="5"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path class="bpt-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <text class="bpt-level" x={8} y={54}>
          Level 1
        </text>
        <text class="bpt-level" x={8} y={138}>
          Level 2
        </text>
        <text class="bpt-level" x={8} y={238}>
          Level 3
        </text>

        <text class="bpt-anno" x={L1.x + L1.w / 2 + 18} y={L1.y + 18}>
          root
        </text>
        <text class="bpt-anno" x={600} y={138} text-anchor="end">
          internal (index)
        </text>
        <text class="bpt-anno" x={600} y={238} text-anchor="end">
          leaf (data entries)
        </text>

        {edges.map((e, i) => (
          <line
            key={`e-${i}`}
            class="bpt-edge"
            x1={e.x1}
            y1={e.y1}
            x2={e.x2}
            y2={e.y2}
            marker-end={`url(#${arrow})`}
          />
        ))}

        {drawNode(L1, true, 'l1')}
        {L2.map((n, i) => drawNode(n, false, `l2-${i}`))}
        {L3.map((n, i) => drawNode(n, false, `l3-${i}`))}
      </svg>
      {caption ? (
        <figcaption class="bpt-caption">
          <MathText text={caption} mathClass="bpt-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const BPlusTreeComponent = BPlusTreeImpl as QuartzMdxComponent<Props>
BPlusTreeComponent.css = style

export const BPlusTree = registerMdxComponent('BPlusTree', BPlusTreeComponent)

export default (() => BPlusTree) satisfies (opts: undefined) => QuartzMdxComponent<Props>
