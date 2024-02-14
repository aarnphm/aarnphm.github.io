import { QuartzComponentConstructor, QuartzComponentProps, QuartzComponent } from "./types"
import style from "./styles/curiusHeader.scss"
//@ts-ignore
import script from "./scripts/curius-search.inline"
import { i18n } from "../i18n"
import { classNames } from "../util/lang"

export default (() => {
  const CuriusTrail: QuartzComponent = (props: QuartzComponentProps) => {
    const { displayClass } = props
    return (
      <div class={classNames(displayClass, "curius-trail")}>
        <ul class="section-ul"></ul>
      </div>
    )
  }

  CuriusTrail.css = style

  return CuriusTrail
}) satisfies QuartzComponentConstructor
