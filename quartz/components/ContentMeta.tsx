import { Date, formatDate, getDate } from "./Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import contentMetaStyle from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"
import readingTime from "reading-time"
import { i18n } from "../i18n"
import { JSX, h } from "preact"

type MetaProp = {
  title: string
  item: JSX.Element
}

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

    const Li = ({ title, item }: MetaProp) => {
      return (
        <li>
          <h3>{title}</h3>
          {item}
        </li>
      )
    }

    const meta = []
    if (created !== undefined) {
      meta.push({
        title: "publié à",
        item: h(
          "span",
          {
            class: "page-creation",
            title: `Date de création du contenu de la page (${created})`,
          },
          [h("em", {}, [<Date date={getDate(cfg, fileData)!} locale={cfg.locale} />])],
        ),
      })
    }
    meta.push(
      { title: "durée", item: h("span", {}, [displayedTime]) },
      {
        title: "source",
        item: h(
          "a",
          {
            href: resolveRelative(fileData.slug!, (fileData.slug! + ".html.md") as FullSlug),
            target: "_blank",
            rel: "noopener noreferrer",
            class: "llm-source",
            style: ["color: inherit", "font-weight: inherit", "text-decoration: underline"].join(
              ";",
            ),
          },
          [h("span", { title: "voir https://github.com/AnswerDotAI/llms-txt" }, ["llms.txt"])],
        ),
      },
    )

    return (
      <ul class={classNames(displayClass, "content-meta")}>
        {meta.map((el) => (
          <Li {...el} />
        ))}
      </ul>
    )
  }

  ContentMeta.css = contentMetaStyle

  return ContentMeta
}) satisfies QuartzComponentConstructor
