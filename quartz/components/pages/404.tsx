import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import styles from "../styles/404.scss"
//@ts-ignore
import notFoundScript from "../scripts/404.inline"

function NotFound(componentData: QuartzComponentProps) {
  return (
    <div class="not-found">
      <article class="popover-hint">
        <h1>404</h1>
        <p>Either this page is private or doesn't exist.</p>
        <div id="typewritter"></div>
      </article>
    </div>
  )
}

NotFound.css = styles
NotFound.afterDOMLoaded = notFoundScript

export default (() => NotFound) satisfies QuartzComponentConstructor
