import { formatDate, getDate } from "./Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import readingTime from "reading-time"
import contentMetaStyle from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"
import { FullSlug, resolveRelative } from "../util/path"

export default (() => {
  const ContentMeta: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    const { text } = fileData

    if (!text) return <></>

    let created: string | undefined
    let modified: string | undefined
    let reading: string | undefined

    if (fileData.dates) {
      created = formatDate(getDate(cfg, fileData)!, cfg.locale)
      modified = formatDate(
        fileData.frontmatter!.modified !== undefined
          ? new Date(fileData.frontmatter!.modified)
          : fileData.dates.modified,
        cfg.locale,
      )
    }

    // Display reading time if enabled
    const { minutes, text: _timeTaken, words: _words } = readingTime(text)
    reading = i18n(cfg.locale).components.contentMeta.readingTime({
      minutes: Math.ceil(minutes),
    })

    return (
      <ul class={classNames(displayClass, "content-meta")}>
        {created !== undefined && (
          <li>
            <span class="page-creation" title="Date de création du contenu de la page">
              <em>{created}</em>
            </span>
          </li>
        )}
        {modified !== undefined && (
          <li>
            <div class="ref-source">
              <span class="page-modification" title="Date de modification du contenu de la page">
                <em>{modified}</em>
              </span>
            </div>
          </li>
        )}
        {reading !== undefined && (
          <li>
            <span className="reading-time" title="Temps de lecture estimé">
              {reading}
            </span>
          </li>
        )}
        <li>
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
