import { QuartzComponentConstructor, QuartzComponentProps } from "../types"

import curiusStyle from "../styles/curiusPage.scss"

//@ts-ignore
import curiusScript from "../scripts/curius.inline"

function CuriusContent(props: QuartzComponentProps) {
  return (
    <div class="popover-hint">
      <div class="curius">
        <p>
          More on{" "}
          <a href="https://curius.app/aaron-pham" target="_blank">
            curius dot app
          </a>
        </p>
        <ul id="curius-container"></ul>
      </div>
      <div class="navigation-container">
        <p>
          You might be interested in{" "}
          <a href="/dump/quotes" class="">
            this
          </a>
        </p>
      </div>
    </div>
  )
}

CuriusContent.css = curiusStyle
CuriusContent.afterDOMLoaded = curiusScript

export default (() => CuriusContent) satisfies QuartzComponentConstructor
