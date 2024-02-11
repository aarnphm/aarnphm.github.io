import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import MetaConstructor from "../Meta"
import NavigationConstructor from "../Navigation"

import style from "../styles/curiusPage.scss"

//@ts-ignore
import script from "../scripts/curius.inline"
import { i18n } from "../../i18n"
import { classNames } from "../../util/lang"

export default (() => {
  function CuriusContent({ displayClass }: QuartzComponentProps) {
    return (
      <div class={classNames(displayClass, "curius", "popover-hint")}>
        <div id="curius-container">
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
  CuriusContent.afterDOMLoaded = script

  return CuriusContent
}) satisfies QuartzComponentConstructor
