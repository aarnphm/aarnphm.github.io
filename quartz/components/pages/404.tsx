import { i18n } from "../../i18n"
import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import styles from "../styles/404.scss"
//@ts-ignore
import notFoundScript from "../scripts/404.inline"

function NotFound({ cfg }: QuartzComponentProps) {
  return (
    <div class="not-found">
      <article class="popover-hint">
        <h1>404</h1>
        <p>{i18n(cfg.locale).pages.error.notFound}</p>
        <div id="typewritter"></div>
      </article>
    </div>
  )
}

NotFound.css = styles
NotFound.afterDOMLoaded = notFoundScript

export default (() => NotFound) satisfies QuartzComponentConstructor
