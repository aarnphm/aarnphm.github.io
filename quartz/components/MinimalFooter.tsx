import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/minimal.scss"
import { version } from "../../package.json"
import { classNames } from "../util/lang"

export default (() => {
  function Footer({ displayClass }: QuartzComponentProps) {
    const year = new Date().getFullYear()

    return (
      <footer class={classNames(displayClass, "minimal-footer")}>
        <div id="year">
          <p>© {year} sur terre</p>
        </div>
        <div id="footnotes">
          <p>
            Vous êtes arrivé au bout! <a href="/">page d'accueil</a>
          </p>
        </div>
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
