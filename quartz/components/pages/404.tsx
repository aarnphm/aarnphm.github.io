import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import styles from "../styles/404.scss"
//@ts-ignore
import notFoundScript from "../scripts/404.inline"

function NotFound(componentData: QuartzComponentProps) {
  return (
    <div id="not-found">
      <article class="popover-hint">
        <a href="/" target="_self">
          <img src="/blob.png" />
        </a>
        <div id="typewritter"></div>
      </article>
    </div>
  )
}

NotFound.css = styles
NotFound.afterDOMLoaded = notFoundScript

export default (() => NotFound) satisfies QuartzComponentConstructor
