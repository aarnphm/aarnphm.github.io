import { MathText } from '../../util/math-text'
import style from '../styles/registerAsStack.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Float = 'none' | 'end'
type Props = { caption?: string; float?: Float }

type Step = { slots: (string | null)[]; sp: number; op: string | null }

const SLOTS = 4

const STEPS: Step[] = [
  { slots: [null, null, null, null], sp: 0, op: null },
  { slots: ['x', null, null, null], sp: 1, op: 's1 := x' },
  { slots: ['x', 'x', null, null], sp: 2, op: 's2 := x' },
  { slots: ['x', 'x', '7', null], sp: 3, op: 's3 := 7' },
  { slots: ['x', 'x + 7', null, null], sp: 2, op: 's2 := s2 + s3' },
  { slots: ['x × (x + 7)', null, null, null], sp: 1, op: 's1 := s1 × s2' },
  { slots: [null, null, null, null], sp: 0, op: 'y := s1' },
]

const RegisterAsStackImpl: QuartzMdxComponent<Props> = ({ caption, float = 'none' }) => {
  const floatClass = float === 'end' ? ' register-as-stack--float-end' : ''

  return (
    <figure class={`register-as-stack${floatClass}`}>
      <div class="ras-grid">
        <fieldset class="ras-stack">
          <legend>Stack (SP = stack pointer)</legend>
          <div class="ras-steps" role="img" aria-label="Register bank used as a stack for y := x × (x + 7).">
            {STEPS.map((step, row) => (
              <div class="ras-row" key={`stack-${row}`} data-ras-sp={step.sp}>
                <div class="ras-slots">
                  <span class="ras-sp" style={{ gridColumn: String(step.sp + 1) }}>
                    ↓ SP
                  </span>
                  {Array.from({ length: SLOTS }, (_, i) => {
                    const value = step.slots[i]
                    const filled = value !== null
                    return (
                      <div
                        class={filled ? 'ras-slot ras-slot--filled' : 'ras-slot'}
                        key={`s-${row}-${i}`}
                      >
                        {filled ? value : ''}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </fieldset>

        <fieldset class="ras-ops">
          <legend>Register Operations</legend>
          <ol class="ras-op-list">
            {STEPS.filter(s => s.op).map((step, i) => (
              <li class="ras-op" key={`op-${i}`}>
                <code>{step.op}</code>
              </li>
            ))}
          </ol>
        </fieldset>
      </div>
      {caption ? (
        <figcaption class="ras-caption">
          <MathText text={caption} mathClass="ras-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const RegisterAsStackComponent = RegisterAsStackImpl as QuartzMdxComponent<Props>
RegisterAsStackComponent.css = style

export const RegisterAsStack = registerMdxComponent('RegisterAsStack', RegisterAsStackComponent)

export default (() => RegisterAsStack) satisfies (opts: undefined) => QuartzMdxComponent<Props>
