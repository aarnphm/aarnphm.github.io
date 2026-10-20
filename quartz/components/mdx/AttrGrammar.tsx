import { MathText } from '../../util/math-text'
import style from '../styles/attrGrammar.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Kind = 'synthesized' | 'inherited'
type Float = 'none' | 'end'
type Props = { caption?: string; kind?: Kind; float?: Float }

type Node = { label: string; kids?: Node[]; lit?: boolean }

const TREES: Record<Kind, Node> = {
  synthesized: {
    label: 'binary(5)',
    lit: true,
    kids: [
      { label: 'binary(2)', kids: [{ label: 'digit(1)' }, { label: 'digit(0)' }] },
      { label: 'digit(1)' },
    ],
  },
  inherited: {
    label: 'expression([], 6)',
    lit: true,
    kids: [
      {
        label: 'term([], 6)',
        kids: [
          {
            label: 'factor([], 6)',
            kids: [{ label: 'integer(2)', kids: [{ label: 'digit(2)' }] }],
          },
        ],
      },
      {
        label: "declaration([], 'a', 3)",
        kids: [
          {
            label: 'expression([], 3)',
            kids: [
              {
                label: 'term([], 3)',
                kids: [
                  {
                    label: 'factor([], 3)',
                    kids: [{ label: 'integer(3)', kids: [{ label: 'digit(3)' }] }],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
}

const renderNode = (node: Node, key: string) => (
  <div class="ag-branch" key={key}>
    <div class={node.lit ? 'ag-node ag-node--lit' : 'ag-node'}>{node.label}</div>
    {node.kids ? (
      <div class="ag-kids">{node.kids.map((k, i) => renderNode(k, `${key}-${i}`))}</div>
    ) : null}
  </div>
)

const AttrGrammarImpl: QuartzMdxComponent<Props> = ({
  caption,
  kind = 'synthesized',
  float = 'none',
}) => {
  const root = TREES[kind] ?? TREES.synthesized
  const floatClass = float === 'end' ? ' attr-grammar--float-end' : ''
  return (
    <figure class={`attr-grammar${floatClass}`} data-ag-kind={kind}>
      <div
        class="ag-root"
        role="img"
        aria-label={
          kind === 'inherited'
            ? 'Inherited attributes flow down the tree.'
            : 'Synthesized attributes flow up the tree.'
        }
      >
        {renderNode(root, 'r')}
      </div>
      {caption ? (
        <figcaption class="ag-caption">
          <MathText text={caption} mathClass="ag-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const AttrGrammarComponent = AttrGrammarImpl as QuartzMdxComponent<Props>
AttrGrammarComponent.css = style

export const AttrGrammar = registerMdxComponent('AttrGrammar', AttrGrammarComponent)

export default (() => AttrGrammar) satisfies (opts: undefined) => QuartzMdxComponent<Props>
