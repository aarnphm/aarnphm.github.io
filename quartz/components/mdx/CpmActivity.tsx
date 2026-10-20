import { MathText } from '../../util/math-text'
import style from '../styles/cpmActivity.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const CpmActivityImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="cpm-activity">
    <div class="cpm-node" role="img" aria-label="CPM activity node template with early and late dates, duration, and float.">
      <div class="cpm-corner cpm-corner--tl">Early Start Date</div>
      <div class="cpm-corner cpm-corner--tr">Early End Date</div>
      <div class="cpm-corner cpm-corner--bl">Late Start Date</div>
      <div class="cpm-corner cpm-corner--br">Late End Date</div>
      <div class="cpm-id">Activity Identifier</div>
      <div class="cpm-name">Activity Name</div>
      <div class="cpm-split">
        <span>Duration</span>
        <span>Float</span>
      </div>
    </div>
    {caption ? (
      <figcaption class="cpm-caption">
        <MathText text={caption} mathClass="cpm-math" />
      </figcaption>
    ) : null}
  </figure>
)

const CpmActivityComponent = CpmActivityImpl as QuartzMdxComponent<Props>
CpmActivityComponent.css = style

export const CpmActivity = registerMdxComponent('CpmActivity', CpmActivityComponent)

export default (() => CpmActivity) satisfies (opts: undefined) => QuartzMdxComponent<Props>
