import { Date, formatDate, getDate } from "./Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import contentMetaStyle from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"
import readingTime from "reading-time"
import { i18n } from "../i18n"

export default (() => {
  const ContentMeta: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    let created: string | undefined

    if (fileData.dates) {
      created = formatDate(getDate(cfg, fileData)!, cfg.locale)
    }
    const { minutes, words: _words } = readingTime(fileData.text!)
    const displayedTime = i18n(cfg.locale).components.contentMeta.readingTime({
      minutes: Math.ceil(minutes),
    })

    return (
      <ul class={classNames(displayClass, "content-meta")}>
        {created !== undefined && (
          <li>
            <h3>publié à</h3>
            <span
              class="page-creation"
              title={`Date de création du contenu de la page (${created})`}
            >
              <em>
                <Date date={getDate(cfg, fileData)!} locale={cfg.locale} />
              </em>
            </span>
          </li>
        )}
        <li>
          <h3>durée</h3>
          <span>{displayedTime}</span>
        </li>
        <li>
          <h3>source</h3>
          <a
            href={resolveRelative(fileData.slug!, (fileData.slug! + ".html.md") as FullSlug)}
            target="_blank"
            rel="noopener noreferrer"
            class="llm-source"
            style={["color: inherit", "font-weight: inherit", "text-decoration: underline"].join(
              ";",
            )}
          >
            <span title="voir https://github.com/AnswerDotAI/llms-txt">llms.txt</span>
          </a>
        </li>
      </ul>
    )
  }

  ContentMeta.css = contentMetaStyle

  return ContentMeta
}) satisfies QuartzComponentConstructor
