import { MathText } from '../../util/math-text'
import style from '../styles/safetyLevels.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const LEVELS = [
  { id: 'ASL-1', text: 'smaller models', tone: '1' },
  { id: 'ASL-2', text: 'present large models', tone: '2' },
  { id: 'ASL-3', text: 'significantly higher risk', tone: '3' },
  { id: 'ASL-4+', text: 'speculative', tone: '4' },
] as const

const SafetyLevelsImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="safety-levels">
    <p class="sl-title">AI safety levels</p>
    <ol class="sl-row">
      {LEVELS.map(level => (
        <li class="sl-card" data-sl-tone={level.tone} key={level.id}>
          <span class="sl-id">{level.id}</span>
          <span class="sl-text">{level.text}</span>
        </li>
      ))}
    </ol>
    <div class="sl-axis" aria-hidden="true">
      <span class="sl-axis-line" />
      <span class="sl-axis-label">
        increasing model capability
        <br />
        increasing security and safety measures
      </span>
    </div>
    {caption ? (
      <figcaption class="sl-caption">
        <MathText text={caption} mathClass="sl-math" />
      </figcaption>
    ) : null}
  </figure>
)

const SafetyLevelsComponent = SafetyLevelsImpl as QuartzMdxComponent<Props>
SafetyLevelsComponent.css = style

export const SafetyLevels = registerMdxComponent('SafetyLevels', SafetyLevelsComponent)

export default (() => SafetyLevels) satisfies (opts: undefined) => QuartzMdxComponent<Props>
