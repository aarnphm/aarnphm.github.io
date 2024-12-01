import { formatDate, getDate } from "./Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import contentMetaStyle from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"

export default (() => {
  const ContentMeta: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    let created: string | undefined

    if (fileData.dates) {
      created = formatDate(getDate(cfg, fileData)!, cfg.locale)
    }

    return (
      <ul class={classNames(displayClass, "content-meta")}>
        {created !== undefined && (
          <li>
            <h3>publié à</h3>
            <span class="page-creation" title="Date de création du contenu de la page">
              <em>{created}</em>
            </span>
          </li>
        )}
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
