import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"
import { version } from "../../package.json"

interface Options {
  links?: Record<string, string>
}

export default ((options?: Options) => {
  function Footer({ cfg, displayClass }: QuartzComponentProps) {
    const year = new Date().getFullYear()
    const links = options?.links ?? []
    return (
      <footer class={`${cfg.defaultFooterStyle}-footer${displayClass ? ` ${displayClass}` : ""}`}>
        {cfg.defaultFooterStyle === "default" ? (
          <>
            <hr />
            <p>
              Built with <a href="https://quartz.jzhao.xyz/">Quartz v{version}</a>, © {year}
            </p>
            <ul>
              {Object.entries(links).map(([text, link]) => (
                <li>
                  <a href={link}>{text}</a>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <>
            <div class="year">
              <p>© {year} on Earth</p>
            </div>
            <div class="footnotes">
              <p>
                Vous êtes arrivé au bout! <a href="/">page d'accueil</a>, avec{" "}
                <a href="https://quartz.jzhao.xyz/" target="_blank">
                  Quartz
                </a>
              </p>
            </div>
          </>
        )}
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
