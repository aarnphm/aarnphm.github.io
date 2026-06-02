import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/tree-reduction.inline'
import style from '../styles/treeReduction.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Leaves = 4 | 8

type Triple = { m: number; z: number; y: number }

type Props = { caption?: string; leaves?: Leaves }

type NodePos = { x: number; y: number; level: number; index: number }

const NODE_W = 92
const NODE_H = 36
const LEVEL_GAP = 92
const PAD_X = 18
const PAD_Y = 24
const SVG_W = 720

const SAMPLE_LEAVES_4: Triple[] = [
  { m: 1.8, z: 2.6, y: 1.4 },
  { m: 2.3, z: 3.1, y: 2.0 },
  { m: 1.2, z: 1.9, y: 0.9 },
  { m: 2.7, z: 3.6, y: 2.5 },
]

const SAMPLE_LEAVES_8: Triple[] = [
  { m: 1.8, z: 2.6, y: 1.4 },
  { m: 2.3, z: 3.1, y: 2.0 },
  { m: 1.2, z: 1.9, y: 0.9 },
  { m: 2.7, z: 3.6, y: 2.5 },
  { m: 1.9, z: 2.8, y: 1.6 },
  { m: 2.5, z: 3.3, y: 2.2 },
  { m: 1.1, z: 1.7, y: 0.7 },
  { m: 2.9, z: 3.9, y: 2.8 },
]

function renderMath(tex: string): string {
  return katex.renderToString(tex, {
    ...katexOptions,
    displayMode: false,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })
}

function buildPositions(leafCount: Leaves): NodePos[][] {
  const levels: NodePos[][] = []
  const rounds = Math.log2(leafCount)
  const totalRows = rounds + 1
  const usableW = SVG_W - PAD_X * 2

  for (let lvl = 0; lvl <= rounds; lvl++) {
    const nodes: NodePos[] = []
    const count = leafCount / 2 ** lvl
    const y = PAD_Y + (totalRows - 1 - lvl) * LEVEL_GAP
    for (let i = 0; i < count; i++) {
      const stride = usableW / count
      const x = PAD_X + stride * (i + 0.5)
      nodes.push({ x, y, level: lvl, index: i })
    }
    levels.push(nodes)
  }
  return levels
}

const Node: FunctionalComponent<{ pos: NodePos; isRoot: boolean }> = ({ pos, isRoot }) => (
  <g
    class={`tr-node ${isRoot ? 'tr-node--root' : ''}`}
    data-tree-node
    data-level={pos.level}
    data-index={pos.index}
    transform={`translate(${pos.x - NODE_W / 2}, ${pos.y - NODE_H / 2})`}
  >
    <rect class="tr-node-shape" x="0" y="0" width={NODE_W} height={NODE_H} rx="6" />
    <foreignObject x="0" y="0" width={NODE_W} height={NODE_H}>
      <div class={`tr-fo ${isRoot ? 'tr-fo--root' : ''}`} data-tree-value>
        -
      </div>
    </foreignObject>
  </g>
)

const stageTex = (level: number, rounds: number): string => {
  if (level === 0) return '\\text{leaves}'
  if (level === rounds) return '\\text{root}'
  return `\\text{stage }${level}`
}

const StageLabel: FunctionalComponent<{ level: number; rounds: number; y: number }> = ({
  level,
  rounds,
  y,
}) => (
  <foreignObject x={SVG_W - PAD_X - 104} y={y - 9} width={100} height={18}>
    <div
      class="tr-fo tr-fo--stage"
      dangerouslySetInnerHTML={{ __html: renderMath(stageTex(level, rounds)) }}
    />
  </foreignObject>
)

const TreeReductionImpl: QuartzMdxComponent<Props> = ({ caption, leaves = 4 }) => {
  const leafCount: Leaves = leaves === 8 ? 8 : 4
  const sample = leafCount === 8 ? SAMPLE_LEAVES_8 : SAMPLE_LEAVES_4
  const positions = buildPositions(leafCount)
  const rounds = positions.length - 1
  const svgH = PAD_Y * 2 + rounds * LEVEL_GAP
  const leavesAttr = JSON.stringify(sample)

  return (
    <figure
      class="tree-reduction"
      data-tree-reduction
      data-leaves={leafCount}
      data-leaves-json={leavesAttr}
    >
      <div class="tr-toolbar">
        <span class="tr-stage" data-tree-stage aria-live="polite">{`stage 0 of ${rounds}`}</span>
        <div class="tr-actions">
          <button
            type="button"
            class="tr-btn tr-btn--primary"
            data-tree-step
            aria-label="Advance reduction by one tree level"
          >
            step stage
          </button>
          <button
            type="button"
            class="tr-btn"
            data-tree-reset
            aria-label="Reset reduction to leaves only"
          >
            reset
          </button>
        </div>
      </div>

      <div class="tr-stage-area">
        <svg
          class="tr-svg"
          viewBox={`0 0 ${SVG_W} ${svgH}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Binary reduction tree over ${leafCount} devices, ${rounds} merge stages`}
        >
          {positions.map((level, lvl) =>
            lvl === 0
              ? null
              : level.map((parent, i) => {
                  const childA = positions[lvl - 1][i * 2]
                  const childB = positions[lvl - 1][i * 2 + 1]
                  return (
                    <g key={`edges-${lvl}-${i}`}>
                      <line
                        class="tr-edge"
                        data-tree-edge
                        data-level={lvl}
                        data-index={i * 2}
                        x1={childA.x}
                        y1={childA.y - NODE_H / 2}
                        x2={parent.x}
                        y2={parent.y + NODE_H / 2}
                      />
                      <line
                        class="tr-edge"
                        data-tree-edge
                        data-level={lvl}
                        data-index={i * 2 + 1}
                        x1={childB.x}
                        y1={childB.y - NODE_H / 2}
                        x2={parent.x}
                        y2={parent.y + NODE_H / 2}
                      />
                    </g>
                  )
                }),
          )}

          {positions.map((level, lvl) => (
            <g key={`row-${lvl}`}>
              <StageLabel level={lvl} rounds={rounds} y={level[0].y} />
              {level.map(pos => (
                <Node
                  key={`n-${lvl}-${pos.index}`}
                  pos={pos}
                  isRoot={lvl === rounds && pos.index === 0}
                />
              ))}
            </g>
          ))}
        </svg>

        <aside class="tr-aside">
          <div class="tr-card">
            <h5>merge rule</h5>
            <p
              dangerouslySetInnerHTML={{
                __html: `Each parent stores ${renderMath('(m, z, y)')} from siblings ${renderMath('a, b')}: ${renderMath('m = \\max(m_a, m_b)')}, ${renderMath('z = z_a e^{m_a - m} + z_b e^{m_b - m}')}, ${renderMath('y = y_a e^{m_a - m} + y_b e^{m_b - m}')}. The root divides ${renderMath('y / z')} for exact softmax.`,
              }}
            />
          </div>

          <div class="tr-card">
            <h5>complexity</h5>
            <div class="tr-complexity">
              <div class="tr-complexity-tree">
                <span>tree</span>
                <span
                  dangerouslySetInnerHTML={{ __html: `${renderMath('\\Theta(\\log p)')} rounds` }}
                />
              </div>
              <div>
                <span>ring</span>
                <span dangerouslySetInnerHTML={{ __html: `${renderMath('\\Theta(p)')} rounds` }} />
              </div>
            </div>
            <p style="margin-top: 0.55rem;">
              Both push <code>p - 1</code> messages; the tree just reorders them into{' '}
              <code>log_2 p</code> latency-bound rounds.
            </p>
          </div>

          <div class="tr-card">
            <h5>intuition</h5>
            <p>
              Once you rescale by the running max, the softmax reduction becomes an associative
              operator on <code>(m, z, y)</code> triples. Any tree topology computes the exact same
              result, no approximation, and the log-depth tree wins on latency. Same trick as
              parallel-prefix scan.
            </p>
          </div>
        </aside>
      </div>

      {caption ? (
        <figcaption class="tr-caption">
          <MathText text={caption} mathClass="tr-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const TreeReductionComponent = TreeReductionImpl as QuartzMdxComponent<Props>
TreeReductionComponent.css = style
TreeReductionComponent.afterDOMLoaded = script

export const TreeReduction = registerMdxComponent('TreeReduction', TreeReductionComponent)

export default (() => TreeReduction) satisfies (opts: undefined) => QuartzMdxComponent<Props>
