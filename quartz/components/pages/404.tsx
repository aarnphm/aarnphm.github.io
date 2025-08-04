import { i18n } from "../../i18n"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"

const NotFound: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
  return (
    <>
      <br />
      <div class="home-tooltip">Cliquez pour retourner à l'accueil</div>
      <article class="popover-hint" data-slug="404">
        <h1>404</h1>
        <p>{i18n(cfg.locale).pages.error.notFound}</p>
      </article>
    </>
  )
}

export default (() => NotFound) satisfies QuartzComponentConstructor
