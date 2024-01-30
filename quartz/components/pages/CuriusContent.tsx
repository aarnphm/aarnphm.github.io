import { QuartzComponentConstructor, QuartzComponentProps } from "../types"

import curiusStyle from "../styles/curiusPage.scss"

//@ts-ignore
import curiusScript from "../scripts/curius.inline"

function CuriusContent(props: QuartzComponentProps) {
  return (
    <div class="popover-hint">
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

CuriusContent.beforeDOMLoaded = `
function getElementBySlug(slug) {
  return document.querySelector(\`body[data-slug="\${slug}"]\`)
}
const slugToCleanUp = ["uses", "dump/quotes", "curius", "influence"]
document.addEventListener('nav', () => {
slugToCleanUp.forEach(slug => {
  if (getElementBySlug(slug)) {
    document
      .querySelector("#quartz-root")
      ?.querySelectorAll(".sidebar")
      .forEach((el) => el.remove())
    document.querySelector("#quartz-root")?.querySelector(".minimal-footer")?.remove()
  }
})
})
`

export default (() => CuriusContent) satisfies QuartzComponentConstructor
