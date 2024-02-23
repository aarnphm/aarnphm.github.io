import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/curiusPage.scss"
//@ts-ignore
import curiusScript from "../scripts/curius.inline"
import { classNames } from "../../util/lang"

export default (() => {
  const CuriusContent: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, "curius", "popover-hint")} id="curius">
        <div class="curius-container">
          <div id="curius-fetching-text"></div>
          <div id="curius-fragments"></div>
          <div class="highlight-modal" id="highlight-modal">
            <ul id="highlight-modal-list"></ul>
          </div>
        </div>
      </div>
    )
  }

  CuriusContent.css = style
  CuriusContent.afterDOMLoaded = curiusScript

  return CuriusContent
}) satisfies QuartzComponentConstructor
