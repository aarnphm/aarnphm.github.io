import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/minimal.scss"
import { i18n } from "../i18n"
import { version } from "../../package.json"
import { classNames } from "../util/lang"

export default (() => {
  function Footer({ displayClass, cfg }: QuartzComponentProps) {
    const year = new Date().getFullYear()

    return (
      <footer class={classNames(displayClass, "minimal-footer")}>
        <div id="year">
          <p>© {year} sur terre</p>
        </div>
        <div id="footnotes">
          <p class="info">
            Vous êtes arrivé au bout! <a href="/">page d'accueil</a>.{" "}
            <span class="desktop-only">
              {i18n(cfg.locale).components.footer.createdWith}{" "}
              <a href="https://quartz.jzhao.xyz/">Quartz v{version}</a>
            </span>
          </p>
        </div>
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
