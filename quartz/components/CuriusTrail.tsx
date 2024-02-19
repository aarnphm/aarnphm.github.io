import { QuartzComponentConstructor, QuartzComponentProps, QuartzComponent } from "./types"
import style from "./styles/curiusTrail.scss"
//@ts-ignore
import script from "./scripts/curius-trail.inline"
import { classNames } from "../util/lang"

const trailLimits = 6

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
  CuriusTrail.afterDOMLoaded = script

  return CuriusTrail
}) satisfies QuartzComponentConstructor
