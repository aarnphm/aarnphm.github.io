import { QuartzComponentConstructor, QuartzComponentProps, QuartzComponent } from "./types"
import style from "./styles/curiusTrail.scss"
import { classNames } from "../util/lang"

const trailLimits = 3

export default (() => {
  const CuriusTrail: QuartzComponent = (props: QuartzComponentProps) => {
    const { cfg, displayClass } = props
    return (
      <div
        class={classNames(displayClass, "curius-trail")}
        data-limits={trailLimits}
        data-locale={cfg.locale}
      >
        <ul class="section-ul" id="trail-list"></ul>
      </div>
    )
  }

  CuriusTrail.css = style

  return CuriusTrail
}) satisfies QuartzComponentConstructor
