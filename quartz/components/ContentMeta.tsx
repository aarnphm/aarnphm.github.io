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
          <h3>{title}</h3>
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
              class: "clipboard-button",
              "data-href": resolveRelative(
                fileData.slug!,
                (fileData.slug! + ".html.md") as FullSlug,
              ),
            },
            [
              h("svg", { ...svgOptions, viewbox: "0 0 16 16", class: "copy-icon" }, [
                h("path", {
                  fillrule: "evenodd",
                  d: "M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z",
                }),
                h("path", {
                  fillrule: "evenodd",
                  d: "M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z",
                }),
              ]),
              h("svg", { ...svgOptions, viewbox: "0 0 16 16", class: "check-icon" }, [
                h("path", {
                  fillrule: "evenodd",
                  fill: "rgb(63, 185, 80)",
                  d: "M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z",
                }),
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
