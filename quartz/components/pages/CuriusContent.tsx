import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { Graph, DarkMode } from "../Landing"

import curiusStyle from "../styles/curiusPage.scss"

//@ts-ignore
import curiusScript from "../scripts/curius.inline"

function CuriusContent(props: QuartzComponentProps) {
  return (
    <div class="popover-hint">
      <Graph />
      <DarkMode />
      <div id="curius">
        <p>
          <span>
            See more on{" "}
            <a href="https://curius.app/aaron-pham" target="_blank">
              curius.app/aaron-pham
            </a>
          </span>
        </p>
        <div class="curius-outer">
          <input
            id="curius-bar"
            type="text"
            aria-label="Search for curius links"
            placeholder="Search for curius links"
          />
          <div id="curius-search-container"></div>
          <div id="curius-container">
            <div class="highlight-modal" id="highlight-modal" style="display: none;">
              <ul id="highlight-modal-list"></ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

CuriusContent.css = curiusStyle
CuriusContent.afterDOMLoaded = curiusScript

export default (() => CuriusContent) satisfies QuartzComponentConstructor
