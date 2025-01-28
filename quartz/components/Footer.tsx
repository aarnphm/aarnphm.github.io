import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"
import { version } from "../../package.json"
import { i18n } from "../i18n"
import { classNames } from "../util/lang"
import { Date as DateComponent, getDate } from "./Date"

type FooterLayout = "default" | "minimal" | "poetry" | "menu" | "curius"

interface Options {
  layout?: FooterLayout
  links?: Record<string, string> &
    Partial<{
      twitter: string
      github: string
      bsky: string
    }>
}

const defaultOptions: Options = { layout: "minimal", links: {} as Record<string, string> }

export default ((userOpts?: Options) => {
  const opts = { ...defaultOptions, ...userOpts }
  const Footer: QuartzComponent = ({ displayClass, cfg, fileData }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const links = opts?.links ?? []

    const DateFooter = () => <DateComponent date={getDate(cfg, fileData)!} locale={cfg.locale} />

    const MinimalFooter = () => (
      <>
        <menu class="icons">
          {Object.entries(links).map(([text, link]) => {
            const label = text.toLowerCase()
            return (
              <li>
                <address>
                  <a href={link} target="_blank" aria-label={`${label}`} title={`${label}`}>
                    {label}
                  </a>
                </address>
              </li>
            )
          })}
          {fileData.frontmatter?.pageLayout! === "letter" && (
            <li>
              <address>
                <a href={"/"} target="_self" class="internal">
                  home
                </a>
              </address>
            </li>
          )}
        </menu>
      </>
    )

    const DefaultFooter = () => (
      <>
        <p>
          {i18n(cfg.locale).components.footer.createdWith}{" "}
          <a href="https://quartz.jzhao.xyz/">Quartz v{version}</a> © {year}
        </p>
        <ul>
          {Object.entries(links).map(([text, link]) => (
            <li>
              <a href={link}>{text}</a>
            </li>
          ))}
        </ul>
      </>
    )

    const FooterConstructor = (layout: FooterLayout) => {
      if (layout === "minimal" || layout === "curius") {
        return <MinimalFooter />
      } else if (layout === "poetry" || layout === "menu") {
        return <DateFooter />
      } else {
        return <DefaultFooter />
      }
    }

    return (
      <footer
        class={classNames(
          displayClass,
          opts.layout!,
          opts.layout !== "curius" ? "title-col" : "curius-col",
        )}
      >
        {FooterConstructor(opts.layout!)}
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
