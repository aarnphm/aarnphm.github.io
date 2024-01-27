import { QuartzComponentConstructor, QuartzComponentProps } from "../types"

import curiusStyle from "../styles/curiusPage.scss"

//@ts-ignore
import curiusScript from "../scripts/curius.inline"

function CuriusContent(props: QuartzComponentProps) {
  return (
    <div class="popover-hint">
      <div id="curius">
        <div id="curius-description"></div>
        <ul id="curius-container"></ul>
      </div>
    </div>
  )
}

CuriusContent.css = curiusStyle
CuriusContent.beforeDOMLoaded = curiusScript

export default (() => CuriusContent) satisfies QuartzComponentConstructor
