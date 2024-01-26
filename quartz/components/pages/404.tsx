import { QuartzComponentConstructor } from "../types"
import styles from "../styles/404.scss"

function NotFound() {
  return (
    <div id="not-found">
      <article class="popover-hint">
        <a href="/" target="_self">
          <img src="/blob.png" />
        </a>
        <p class="not-found-title">oops! this page is still in the oven!</p>
        <a href="/" class="not-found-button">
          <p>page d'accueil</p>
        </a>
      </article>
    </div>
  )
}

NotFound.css = styles

export default (() => NotFound) satisfies QuartzComponentConstructor
