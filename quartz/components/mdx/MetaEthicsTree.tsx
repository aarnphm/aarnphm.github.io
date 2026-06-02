import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/meta-ethics-tree.inline'
import style from '../styles/metaEthicsTree.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type Kind = 'question' | 'category' | 'terminal'

type Node = {
  id: string
  kind: Kind
  header?: string
  text: string
  qshort?: string
  x: number
  y: number
  w: number
  h: number
  rx: number
  sig?: string
  gloss?: string
  mine?: boolean
}

const tex = (t: string): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: false,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const M: FunctionalComponent<{ t: string; cls?: string }> = ({ t, cls }) => (
  <span class={cls} dangerouslySetInnerHTML={{ __html: tex(t) }} />
)

const DEFAULT = 'obj'

const NODES: Node[] = [
  {
    id: 'q0',
    kind: 'question',
    text: 'can a moral claim be true or false?',
    qshort: 'truth-apt?',
    x: 354,
    y: 18,
    w: 172,
    h: 54,
    rx: 3,
  },
  {
    id: 'cog',
    kind: 'category',
    header: 'cognitivism',
    text: 'are any moral claims true?',
    qshort: 'any true?',
    x: 162,
    y: 128,
    w: 176,
    h: 84,
    rx: 3,
    sig: 'v(p)\\in\\{\\top,\\bot\\}',
    gloss: 'moral claims express beliefs that can be true or false.',
  },
  {
    id: 'noncog',
    kind: 'category',
    header: 'non-cognitivism',
    text: 'what do moral claims express?',
    qshort: 'expresses?',
    x: 556,
    y: 128,
    w: 176,
    h: 84,
    rx: 3,
    sig: 'v(p)=\\varnothing',
    gloss: 'moral claims voice attitudes, so they are not true or false.',
  },
  {
    id: 'realism',
    kind: 'category',
    header: 'moral realism',
    text: 'is moral truth independent of belief?',
    qshort: 'belief-independent?',
    x: 36,
    y: 268,
    w: 176,
    h: 84,
    rx: 3,
    sig: '\\exists\\,p:\\ v(p)=\\top',
    gloss: 'at least some moral claims come out true.',
  },
  {
    id: 'err',
    kind: 'terminal',
    text: 'error theory',
    x: 260,
    y: 280,
    w: 140,
    h: 40,
    rx: 5,
    sig: '\\forall\\,p:\\ v(p)=\\bot',
    gloss: 'moral claims are truth-apt yet uniformly false.',
  },
  {
    id: 'emot',
    kind: 'terminal',
    text: 'emotivism',
    x: 490,
    y: 280,
    w: 140,
    h: 40,
    rx: 5,
    sig: 'p\\mapsto\\text{emotion}',
    gloss: 'moral talk vents feeling (boo murder, hooray charity).',
  },
  {
    id: 'univ',
    kind: 'terminal',
    text: 'universal prescriptivism',
    x: 668,
    y: 278,
    w: 152,
    h: 46,
    rx: 5,
    sig: 'p\\mapsto\\text{imperative}',
    gloss: 'moral claims are imperatives meant to bind everyone.',
  },
  {
    id: 'obj',
    kind: 'terminal',
    text: 'moral objectivism',
    x: 8,
    y: 430,
    w: 152,
    h: 46,
    rx: 5,
    sig: 'v(p)\\perp\\text{belief}',
    gloss: 'moral truth holds independently of what anyone believes.',
    mine: true,
  },
  {
    id: 'q4',
    kind: 'question',
    text: 'what determines moral truth?',
    qshort: 'fixed by?',
    x: 214,
    y: 424,
    w: 172,
    h: 54,
    rx: 3,
  },
  {
    id: 'subj',
    kind: 'terminal',
    text: 'subjectivism',
    x: 120,
    y: 540,
    w: 140,
    h: 40,
    rx: 5,
    sig: 'v(p)=f(\\text{self})',
    gloss: 'each individual’s stance fixes moral truth.',
  },
  {
    id: 'rel',
    kind: 'terminal',
    text: 'relativism',
    x: 280,
    y: 540,
    w: 140,
    h: 40,
    rx: 5,
    sig: 'v(p)=f(\\text{culture})',
    gloss: 'a society or culture fixes moral truth.',
  },
  {
    id: 'divine',
    kind: 'terminal',
    text: 'divine command theory',
    x: 435,
    y: 540,
    w: 152,
    h: 46,
    rx: 5,
    sig: 'v(p)=f(\\text{God})',
    gloss: 'god’s commands fix moral truth.',
  },
]

const EDGES_RAW: Array<[string, string, string]> = [
  ['q0', 'cog', 'yes'],
  ['q0', 'noncog', 'no'],
  ['cog', 'realism', 'yes'],
  ['cog', 'err', 'no'],
  ['noncog', 'emot', 'raw emotions'],
  ['noncog', 'univ', 'imperatives'],
  ['realism', 'obj', 'yes'],
  ['realism', 'q4', 'no'],
  ['q4', 'subj', 'individuals'],
  ['q4', 'rel', 'society'],
  ['q4', 'divine', 'god'],
]

const byId = (id: string): Node => NODES.find(n => n.id === id)!
const PARENT: Record<string, [string, string]> = {}
for (const [from, to, label] of EDGES_RAW) PARENT[to] = [from, label]

const ek = (from: string, to: string) => `${from}->${to}`

const pathNodes = (id: string): string[] => {
  const arr = [id]
  let cur = id
  while (PARENT[cur]) {
    cur = PARENT[cur][0]
    arr.unshift(cur)
  }
  return arr
}

const pathEdges = (nodes: string[]): string[] => {
  const out: string[] = []
  for (let i = 1; i < nodes.length; i++) out.push(ek(nodes[i - 1], nodes[i]))
  return out
}

const answersFor = (nodes: string[]): Array<{ q: string; a: string }> => {
  const out: Array<{ q: string; a: string }> = []
  for (let i = 1; i < nodes.length; i++) {
    out.push({ q: byId(nodes[i - 1]).qshort ?? '', a: PARENT[nodes[i]][1] })
  }
  return out
}

const topC = (n: Node) => ({ x: n.x + n.w / 2, y: n.y })
const botC = (n: Node) => ({ x: n.x + n.w / 2, y: n.y + n.h })
const vcurve = (a: { x: number; y: number }, b: { x: number; y: number }): string => {
  const m = (a.y + b.y) / 2
  return `M ${a.x} ${a.y} C ${a.x} ${m}, ${b.x} ${m}, ${b.x} ${b.y}`
}

const VW = 840
const VH = 600

const SELECTABLE = NODES.filter(n => n.kind !== 'question')

const NodeGroup: FunctionalComponent<{ node: Node }> = ({ node }) => {
  const selectable = node.kind !== 'question'
  const nodes = selectable ? pathNodes(node.id) : []
  const aria = `${node.header ? `${node.header}: ` : ''}${node.text}`
  return (
    <g
      class={`mep-node mep-node--${node.kind}${selectable ? ' mep-node--selectable' : ''}`}
      data-mep-node={node.id}
      data-mep-selectable={selectable ? 'true' : undefined}
      data-mep-path={selectable ? nodes.join(',') : undefined}
      data-mep-edges={selectable ? pathEdges(nodes).join(',') : undefined}
      transform={`translate(${node.x}, ${node.y})`}
      role={selectable ? 'button' : 'img'}
      tabindex={selectable ? 0 : undefined}
      aria-label={aria}
    >
      <rect class="mep-box" width={node.w} height={node.h} rx={node.rx} />
      <foreignObject width={node.w} height={node.h}>
        <div class={`mep-fo mep-fo--${node.kind}`}>
          {node.header ? <span class="mep-cat-name">{node.header}</span> : null}
          {node.kind === 'terminal' ? (
            <span class="mep-pos-name">{node.text}</span>
          ) : (
            <span class="mep-q-text">{node.text}</span>
          )}
        </div>
      </foreignObject>
      {node.mine ? <circle class="mep-you" cx={node.w - 9} cy={9} r={4} /> : null}
    </g>
  )
}

const EdgeGroup: FunctionalComponent<{ from: string; to: string; label: string }> = ({
  from,
  to,
  label,
}) => {
  const a = botC(byId(from))
  const b = topC(byId(to))
  const mx = (a.x + b.x) / 2
  const my = (a.y + b.y) / 2
  const lw = Math.max(40, label.length * 6.4 + 12)
  return (
    <g class="mep-edge-group" data-mep-edge={ek(from, to)}>
      <path class="mep-edge" d={vcurve(a, b)} />
      <foreignObject class="mep-edge-label" x={mx - lw / 2} y={my - 9} width={lw} height={18}>
        <div class="mep-edge-text">{label}</div>
      </foreignObject>
    </g>
  )
}

const Detail: FunctionalComponent<{ node: Node }> = ({ node }) => (
  <div class="mep-detail" data-mep-detail={node.id}>
    <h4 class="mep-detail-name">
      {node.header ?? node.text}
      {node.mine ? <span class="mep-mine-tag">my view</span> : null}
    </h4>
    <M t={node.sig ?? ''} cls="mep-sig" />
    <p class="mep-gloss">{node.gloss}</p>
  </div>
)

const PathCard: FunctionalComponent<{ node: Node }> = ({ node }) => {
  const answers = answersFor(pathNodes(node.id))
  return (
    <ol class="mep-answers" data-mep-detail={node.id} role="list">
      {answers.map(a => (
        <li class="mep-answer">
          <span class="mep-answer-q">{a.q}</span>
          <span class="mep-answer-a">{a.a}</span>
        </li>
      ))}
    </ol>
  )
}

const MetaEthicsTreeImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="meta-ethics-tree" data-meta-ethics data-mep-default={DEFAULT}>
    <div class="mep-stage">
      <div class="mep-canvas">
        <svg
          class="mep-svg"
          viewBox={`0 0 ${VW} ${VH}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Meta-ethical positions decision tree: a root question about whether moral claims can be true or false branches through cognitivism and non-cognitivism into ten terminal positions; selecting a position highlights the chain of answers leading to it, defaulting to moral objectivism."
        >
          {EDGES_RAW.map(([from, to, label]) => (
            <EdgeGroup key={ek(from, to)} from={from} to={to} label={label} />
          ))}
          {NODES.map(n => (
            <NodeGroup key={n.id} node={n} />
          ))}
        </svg>
      </div>

      <aside class="mep-side" aria-label="Selected position">
        <div class="mep-card mep-card--detail">
          {SELECTABLE.map(n => (
            <Detail key={n.id} node={n} />
          ))}
        </div>
        <div class="mep-card mep-card--path">
          <h4>decision path</h4>
          {SELECTABLE.map(n => (
            <PathCard key={n.id} node={n} />
          ))}
        </div>
        <button type="button" class="mep-reset" data-mep-reset aria-label="Return to my position">
          back to my view
        </button>
      </aside>
    </div>

    {caption ? (
      <figcaption class="mep-caption">
        <MathText text={caption} mathClass="mep-math" />
      </figcaption>
    ) : null}
  </figure>
)

const MetaEthicsTreeComponent = MetaEthicsTreeImpl as QuartzMdxComponent<Props>
MetaEthicsTreeComponent.css = style
MetaEthicsTreeComponent.afterDOMLoaded = script

export const MetaEthicsTree = registerMdxComponent('MetaEthicsTree', MetaEthicsTreeComponent)

export default (() => MetaEthicsTree) satisfies (opts: undefined) => QuartzMdxComponent<Props>
