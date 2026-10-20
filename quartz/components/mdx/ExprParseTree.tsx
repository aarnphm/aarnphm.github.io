import { MathText } from '../../util/math-text'
import style from '../styles/exprParseTree.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type TreeId =
  | 'id+id'
  | 'id+id+id'
  | 'id+left'
  | 'id+right'
  | 'id+id*id'
  | 'id+*+*'
  | 'idparen'
  | 'idast'
  | 'nl'
type Float = 'none' | 'end'
type Props = { caption?: string; tree?: TreeId; float?: Float }

type Node = { label: string; kids?: Node[]; lit?: boolean }

const TREES: Record<TreeId, Node> = {
  'id+id': { label: '+', lit: true, kids: [{ label: 'id' }, { label: 'id' }] },
  'id+id+id': {
    label: '+',
    lit: true,
    kids: [{ label: '+', kids: [{ label: 'id' }, { label: 'id' }] }, { label: 'id' }],
  },
  'id+left': {
    label: '+',
    lit: true,
    kids: [{ label: '+', kids: [{ label: 'id' }, { label: 'id' }] }, { label: 'id' }],
  },
  'id+right': {
    label: '+',
    lit: true,
    kids: [{ label: 'id' }, { label: '+', kids: [{ label: 'id' }, { label: 'id' }] }],
  },
  'id+id*id': {
    label: '+',
    lit: true,
    kids: [{ label: 'id' }, { label: '×', kids: [{ label: 'id' }, { label: 'id' }] }],
  },
  'id+*+*': {
    label: '+',
    lit: true,
    kids: [
      { label: '×', kids: [{ label: 'id' }, { label: 'id' }] },
      { label: '×', kids: [{ label: 'id' }, { label: 'id' }] },
    ],
  },
  idparen: {
    label: '+',
    lit: true,
    kids: [{ label: 'id' }, { label: '(…)', kids: [{ label: '+', kids: [{ label: 'id' }, { label: 'id' }] }] }],
  },
  idast: {
    label: ':=',
    lit: true,
    kids: [
      { label: 'id' },
      { label: '+', kids: [{ label: 'id' }, { label: '×', kids: [{ label: 'id' }, { label: 'num' }] }] },
    ],
  },
  nl: {
    label: 'S',
    lit: true,
    kids: [{ label: 'NP', kids: [{ label: 'Det' }, { label: 'N' }] }, { label: 'VP', kids: [{ label: 'V' }, { label: 'NP' }] }],
  },
}

const renderNode = (node: Node, key: string) => (
  <div class="ept-branch" key={key}>
    <div class={node.lit ? 'ept-node ept-node--lit' : 'ept-node'}>{node.label}</div>
    {node.kids ? <div class="ept-kids">{node.kids.map((k, i) => renderNode(k, `${key}-${i}`))}</div> : null}
  </div>
)

const ExprParseTreeImpl: QuartzMdxComponent<Props> = ({
  caption,
  tree = 'id+id',
  float = 'none',
}) => {
  const root = TREES[tree] ?? TREES['id+id']
  const floatClass = float === 'end' ? ' expr-parse-tree--float-end' : ''
  return (
    <figure class={`expr-parse-tree${floatClass}`}>
      <div class="ept-root" role="img" aria-label={`Parse tree ${tree}`}>
        {renderNode(root, 'r')}
      </div>
      {caption ? (
        <figcaption class="ept-caption">
          <MathText text={caption} mathClass="ept-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ExprParseTreeComponent = ExprParseTreeImpl as QuartzMdxComponent<Props>
ExprParseTreeComponent.css = style

export const ExprParseTree = registerMdxComponent('ExprParseTree', ExprParseTreeComponent)

export default (() => ExprParseTree) satisfies (opts: undefined) => QuartzMdxComponent<Props>
