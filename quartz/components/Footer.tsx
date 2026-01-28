import { version, repository } from "../../package.json"
import { i18n } from "../i18n"
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import { classNames } from "../util/lang"
import { Date as DateComponent, getDate } from "./Date"
import style from "./styles/footer.scss"

type FooterLayout = "default" | "minimal" | "poetry" | "menu" | "curius" | "masonry"

interface Options {
  layout?: FooterLayout
  links?: Record<string, string> & Partial<{ twitter: string; github: string; bsky: string }>
}

const defaultOptions: Options = { layout: "minimal", links: {} as Record<string, string> }

export default ((userOpts?: Options) => {
  const opts = { ...defaultOptions, ...userOpts }
  const Footer: QuartzComponent = ({ displayClass, cfg, fileData, ctx }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const links = opts?.links ?? []
    const addHomeLink =
      (fileData.frontmatter && fileData.frontmatter.pageLayout! === "letter") ||
      fileData.slug === "curius"

    const DateFooter = () => <DateComponent date={getDate(cfg, fileData)!} locale={cfg.locale} />

    const Sha = () => {
      const fullSha =
        process.env.WORKERS_CI_COMMIT_SHA ||
        process.env.CF_PAGES_COMMIT_SHA ||
        process.env.GITHUB_SHA ||
        ctx?.gitCommitSha ||
        ""
      if (!fullSha) return null
      const shortSha = fullSha.slice(0, 7)
      const repoUrl = (repository?.url || "").replace(/\.git$/, "")
      const commitUrl = repoUrl ? `${repoUrl}/commit/${fullSha}` : undefined
      return (
        <>
          {", "}
          {commitUrl ? (
            <a
              href={commitUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`commit ${shortSha}`}
              style="color: var(--tertiary)"
            >
              {shortSha}
            </a>
          ) : (
            <span>{shortSha}</span>
          )}
        </>
      )
    }

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
          {addHomeLink && (
            <li>
              <address>
                <a href={"/"} target="_self" class="internal">
                  home
                </a>
              </address>
            </li>
          )}
        </menu>
        <p>
          <a
            href="https://quartz.jzhao.xyz/"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Quartz links"
          >
            Quartz v{version}
          </a>{" "}
          © {year}
          <Sha />
        </p>
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
      } else if (layout === "masonry") {
        return <></>
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
