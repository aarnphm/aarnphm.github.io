import { MathText } from '../../util/math-text'
import style from '../styles/compilerPipeline.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Diagram =
  | 'task'
  | 'phases'
  | 'phases-example'
  | 'passes'
  | 'frontend-backend'
  | 'bytecode'
  | 'asm-pipeline'
type Float = 'none' | 'end'
type Props = { caption?: string; diagram?: Diagram; float?: Float }

const PHASES = [
  { name: 'lexical analysis', out: 'sequence of symbols' },
  { name: 'syntactic analysis', out: 'syntax tree' },
  { name: 'contextual analysis', out: 'syntax tree with context' },
  { name: 'intermediate code generation', out: 'intermediate code' },
  { name: 'code optimization', out: 'intermediate code' },
  { name: 'target code generation', out: 'target code' },
] as const

const PASSES = [
  { name: 'lexical analysis', kind: 'analysis' },
  { name: 'syntactic and contextual analysis', kind: 'analysis' },
  { name: 'synthesis', kind: 'synthesis' },
] as const

const FRONTENDS = ['C', 'Java', 'Go'] as const
const BACKENDS = ['RISC-V', 'ARMv8', 'x86-64'] as const

const CompilerPipelineImpl: QuartzMdxComponent<Props> = ({
  caption,
  diagram = 'phases',
  float = 'none',
}) => {
  const floatClass = float === 'end' ? ' compiler-pipeline--float-end' : ''

  return (
    <figure class={`compiler-pipeline${floatClass}`} data-cp-diagram={diagram}>
      {diagram === 'task' ? (
        <div class="cp-task" role="img" aria-label="Compiler turns a source program into target code or error messages.">
          <div class="cp-box">source program</div>
          <span class="cp-arrow" aria-hidden="true">
            →
          </span>
          <div class="cp-box cp-box--lit">compiler</div>
          <div class="cp-task-outs">
            <div class="cp-box">target code</div>
            <div class="cp-box">error messages</div>
          </div>
        </div>
      ) : null}

      {diagram === 'phases' ? (
        <ol class="cp-phases" role="img" aria-label="Compiler phases from source text to target code.">
          <li class="cp-io">source text</li>
          {PHASES.map((p, i) => (
            <li class="cp-phase" key={p.name} data-cp-band={i < 3 ? 'analysis' : 'synthesis'}>
              <span class="cp-phase-name">{p.name}</span>
              <span class="cp-phase-out">{p.out}</span>
            </li>
          ))}
        </ol>
      ) : null}

      {diagram === 'phases-example' ? (
        <div
          class="cp-example"
          role="img"
          aria-label="Compilation of dist := dist + rot × 24 through lexical tokens, a syntax tree, and typed intermediate code."
        >
          <div class="cp-ex-col">
            <p class="cp-ex-label">source</p>
            <pre class="cp-ex-code">{`dist := dist + rot × 24`}</pre>
          </div>
          <div class="cp-ex-col">
            <p class="cp-ex-label">tokens</p>
            <p class="cp-ex-toks">id := id + id × number</p>
          </div>
          <div class="cp-ex-col cp-ex-col--tree">
            <p class="cp-ex-label">syntax tree</p>
            <div class="cp-tree">
              <div class="cp-tree-node cp-tree-node--lit">:=</div>
              <div class="cp-tree-row">
                <div class="cp-tree-node">dist</div>
                <div class="cp-tree-branch">
                  <div class="cp-tree-node">+</div>
                  <div class="cp-tree-row">
                    <div class="cp-tree-node">dist</div>
                    <div class="cp-tree-branch">
                      <div class="cp-tree-node">×</div>
                      <div class="cp-tree-row">
                        <div class="cp-tree-node">rot</div>
                        <div class="cp-tree-node">24</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="cp-ex-col">
            <p class="cp-ex-label">with context</p>
            <p class="cp-ex-note">all expressions typed integer</p>
          </div>
        </div>
      ) : null}

      {diagram === 'passes' ? (
        <ol class="cp-passes" role="img" aria-label="Compiler passes merge several phases.">
          {PASSES.map(p => (
            <li class="cp-pass" key={p.name} data-cp-band={p.kind}>
              {p.name}
            </li>
          ))}
        </ol>
      ) : null}

      {diagram === 'frontend-backend' || diagram === 'bytecode' ? (
        <div
          class="cp-fb"
          role="img"
          aria-label={
            diagram === 'bytecode'
              ? 'Front ends emit byte code for back ends and an interpreter.'
              : 'Language front ends share a syntax tree with context across target back ends.'
          }
        >
          <div class="cp-fb-col">
            {FRONTENDS.map(lang => (
              <div class="cp-box cp-box--lit" key={lang}>
                {lang} front end
              </div>
            ))}
          </div>
          <div class="cp-box cp-box--mid">
            {diagram === 'bytecode' ? 'byte code' : 'syntax tree with context'}
          </div>
          <div class="cp-fb-col">
            {BACKENDS.map(t => (
              <div class="cp-box" key={t}>
                {t} back end
              </div>
            ))}
            {diagram === 'bytecode' ? <div class="cp-box">interpreter</div> : null}
          </div>
        </div>
      ) : null}

      {diagram === 'asm-pipeline' ? (
        <ol class="cp-passes" role="img" aria-label="High-level language to assembly to machine code.">
          <li class="cp-pass" data-cp-band="analysis">
            high-level language program
          </li>
          <li class="cp-io">compiler</li>
          <li class="cp-pass" data-cp-band="analysis">
            assembly language program
          </li>
          <li class="cp-io">assembler</li>
          <li class="cp-pass" data-cp-band="synthesis">
            machine language program
          </li>
          <li class="cp-io">machine interpretation</li>
        </ol>
      ) : null}

      {caption ? (
        <figcaption class="cp-caption">
          <MathText text={caption} mathClass="cp-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const CompilerPipelineComponent = CompilerPipelineImpl as QuartzMdxComponent<Props>
CompilerPipelineComponent.css = style

export const CompilerPipeline = registerMdxComponent(
  'CompilerPipeline',
  CompilerPipelineComponent,
)

export default (() => CompilerPipeline) satisfies (opts: undefined) => QuartzMdxComponent<Props>
