import { MathText } from '../../util/math-text'
import style from '../styles/grammarDiagram.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Kind =
  | 'terminal'
  | 'nonterminal'
  | 'sequence'
  | 'choice'
  | 'option'
  | 'repetition'
  | 'parenthesis'
  | 'production'
  | 'railroad'
type Float = 'none' | 'end'
type Props = { caption?: string; kind?: Kind; float?: Float }

const LABELS: Record<Kind, string> = {
  terminal: 'terminal',
  nonterminal: 'nonterminal',
  sequence: 'sequence',
  choice: 'choice',
  option: 'option',
  repetition: 'repetition',
  parenthesis: 'grouping',
  production: 'production',
  railroad: 'railroad diagram',
}

const GrammarDiagramImpl: QuartzMdxComponent<Props> = ({
  caption,
  kind = 'terminal',
  float = 'none',
}) => {
  const floatClass = float === 'end' ? ' grammar-diagram--float-end' : ''
  return (
    <figure class={`grammar-diagram${floatClass}`} data-gd-kind={kind}>
      <div class="gd-canvas" role="img" aria-label={LABELS[kind]}>
        {kind === 'terminal' ? <span class="gd-term">terminal</span> : null}
        {kind === 'nonterminal' ? <span class="gd-nonterm">Nonterminal</span> : null}
        {kind === 'sequence' ? (
          <div class="gd-row">
            <span class="gd-term">a</span>
            <span class="gd-rail" />
            <span class="gd-term">b</span>
            <span class="gd-rail" />
            <span class="gd-term">c</span>
          </div>
        ) : null}
        {kind === 'choice' ? (
          <div class="gd-choice">
            <span class="gd-term">a</span>
            <span class="gd-term">b</span>
            <span class="gd-term">c</span>
          </div>
        ) : null}
        {kind === 'option' ? (
          <div class="gd-option">
            <span class="gd-bypass" />
            <span class="gd-term">a</span>
          </div>
        ) : null}
        {kind === 'repetition' ? (
          <div class="gd-rep">
            <span class="gd-term">a</span>
            <span class="gd-loop">↻</span>
          </div>
        ) : null}
        {kind === 'parenthesis' ? (
          <div class="gd-row">
            <span class="gd-paren">(</span>
            <span class="gd-term">a</span>
            <span class="gd-rail" />
            <span class="gd-term">b</span>
            <span class="gd-paren">)</span>
          </div>
        ) : null}
        {kind === 'production' ? (
          <div class="gd-row">
            <span class="gd-nonterm">A</span>
            <span class="gd-arrow">→</span>
            <span class="gd-term">a</span>
            <span class="gd-rail" />
            <span class="gd-nonterm">B</span>
          </div>
        ) : null}
        {kind === 'railroad' ? (
          <div class="gd-railroad">
            <span class="gd-nonterm">Expr</span>
            <span class="gd-arrow">→</span>
            <div class="gd-choice">
              <span class="gd-term">id</span>
              <div class="gd-row">
                <span class="gd-term">(</span>
                <span class="gd-nonterm">Expr</span>
                <span class="gd-term">)</span>
              </div>
            </div>
          </div>
        ) : null}
      </div>
      {caption ? (
        <figcaption class="gd-caption">
          <MathText text={caption} mathClass="gd-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const GrammarDiagramComponent = GrammarDiagramImpl as QuartzMdxComponent<Props>
GrammarDiagramComponent.css = style

export const GrammarDiagram = registerMdxComponent('GrammarDiagram', GrammarDiagramComponent)

export default (() => GrammarDiagram) satisfies (opts: undefined) => QuartzMdxComponent<Props>
