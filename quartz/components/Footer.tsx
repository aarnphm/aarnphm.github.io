import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"
import { version } from "../../package.json"

interface Options {
  style?: "default" | "minimal"
  disable?: boolean
  links?: Record<string, string>
}

const defaultOptions: Options = {
  style: "minimal",
  disable: false,
}

export default ((options?: Options) => {
  const opts = { ...defaultOptions, ...options }
  function Footer({ displayClass }: QuartzComponentProps) {
    const year = new Date().getFullYear()
    const links = opts?.links ?? []
    if (opts?.disable) return <></>
    return opts.style === "default" ? (
      <footer class={`${opts.style}-footer${displayClass ?? ""}`}>
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
      </footer>
    ) : (
      <footer class={`${opts.style}-footer${displayClass ?? ""}`}>
        <div class="year">
          <p>© {year} on Earth</p>
        </div>
        <div class="footnotes">
          <p>
            Vous êtes arrivé au bout! <a href="/">Retour</a>
          </p>
        </div>
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
