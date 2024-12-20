import { Date as DateComponent, formatDate, getDate } from "./Date"
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
  classes: string[]
  item: JSX.Element | JSX.Element[]
}

export default (() => {
  const ContentMeta: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    let created: Date | undefined
    let modified: Date | undefined

    if (fileData.dates) {
      created = getDate(cfg, fileData)
    }
    if (fileData.dates?.modified) {
      modified = fileData.dates?.["modified"]
    }
    const { minutes, words: _words } = readingTime(fileData.text!)
    const displayedTime = i18n(cfg.locale).components.contentMeta.readingTime({
      minutes: Math.ceil(minutes),
    })

    const Li = ({ title, item, classes }: MetaProp) => {
      return (
        <li class={classNames(undefined, ...classes)}>
          <h2>{title}</h2>
          <div class="container">{item}</div>
        </li>
      )
    }

    const meta: MetaProp[] = []
    if (created !== undefined) {
      meta.push({
        title: "publié à",
        classes: ["published-time"],
        item: h(
          "span",
          {
            class: "page-creation",
            title: `Date de création du contenu de la page (${created})`,
          },
          [h("em", {}, [<DateComponent date={created} locale={cfg.locale} />])],
        ),
      })
    }
    if (modified !== undefined) {
      meta.push({
        title: "modifié à",
        classes: ["modified-time"],
        item: h("span", { class: "page-modification" }, [
          h("em", {}, [<DateComponent date={modified} locale={cfg.locale} />]),
        ]),
      })
    }
    meta.push(
      { title: "durée", classes: ["reading-time"], item: h("span", {}, [displayedTime]) },
      {
        title: "source",
        classes: ["readable-source"],
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
              ariaHidden: true,
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
