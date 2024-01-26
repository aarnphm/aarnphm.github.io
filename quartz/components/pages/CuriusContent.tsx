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
CuriusContent.afterDOMLoaded = curiusScript

CuriusContent.beforeDOMLoaded = `
function getElementBySlug(slug) {
  return document.querySelector(\`body[data-slug="\${slug}"]\`)
}
const slugToCleanUp = ["uses", "dump/quotes", "curius", "influence"]
document.addEventListener('nav', () => {
slugToCleanUp.forEach(slug => {
  const el = getElementBySlug(slug)
  console.log(el)
  if (el) {
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
