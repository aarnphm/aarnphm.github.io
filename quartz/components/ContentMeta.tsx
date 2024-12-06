import { Date, formatDate, getDate } from "./Date"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
//@ts-ignore
import script from "./scripts/content-meta.inline"
import style from "./styles/contentMeta.scss"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"
import readingTime from "reading-time"
import { i18n } from "../i18n"
import { JSX, h } from "preact"
import { svgOptions } from "./renderPage"

type MetaProp = {
  title: string
  item: JSX.Element | JSX.Element[]
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
          <h2>{title}</h2>
          <div class="container">{item}</div>
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
        item: [
          h(
            "a",
            {
              href: resolveRelative(fileData.slug!, (fileData.slug! + ".html.md") as FullSlug),
              target: "_blank",
              rel: "noopener noreferrer",
              class: "llm-source",
            },
            [h("span", { title: "voir https://github.com/AnswerDotAI/llms-txt" }, ["llms.txt"])],
          ),
          h(
            "span",
            {
              type: "button",
              arialabel: "copy source",
              tabindex: -1,
              ariahidden: true,
              class: "clipboard-button",
              "data-href": resolveRelative(
                fileData.slug!,
                (fileData.slug! + ".html.md") as FullSlug,
              ),
            },
            [
              h("svg", { ...svgOptions, viewbox: "0 -8 24 24", class: "copy-icon" }, [
                h("use", { href: "#github-copy" }),
              ]),
              h("svg", { ...svgOptions, viewbox: "0 -8 24 24", class: "check-icon" }, [
                h("use", { href: "#github-check" }),
              ]),
            ],
          ),
        ],
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

  ContentMeta.css = style
  ContentMeta.afterDOMLoaded = script

  return ContentMeta
}) satisfies QuartzComponentConstructor
