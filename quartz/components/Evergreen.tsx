import { QuartzComponent, QuartzComponentProps } from "./types"
import { getAllSegmentPrefixes, resolveRelative, simplifySlug } from "../util/path"
import { h, VNode } from "preact"
import { i18n } from "../i18n"
//@ts-ignore
import script from "./scripts/evergreen.inline"
import style from "./styles/evergreen.scss"
import { unescapeHTML } from "../util/escape"
import { QuartzPluginData } from "../plugins/vfile"

type Props = {
  vaults?: QuartzPluginData[]
  content?: VNode
} & QuartzComponentProps

export const AllTags: QuartzComponent = ({ cfg, allFiles }: Props) => {
  const tags = [
    ...new Set(
      allFiles.flatMap((data) => data.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes),
    ),
  ].sort((a, b) => a.localeCompare(b))

  return h("section", { class: "note-tags" }, [
    h("h3", { class: "note-subtitle" }, [i18n(cfg.locale).pages.tagContent.tag]),
    h(
      "div",
      { class: "notes-list" },
      tags.map((tag) => h("div", { class: "note-tag", "data-tag": tag }, [tag])),
    ),
  ])
}

const EvergreenNotes = ({ cfg, fileData, vaults }: Props) => {
  const larges = ["thoughts/mechanistic-interpretability", "thoughts/vllm"]
  const smalls = [
    "thoughts/constrained-decoding",
    "thoughts/LLMs",
    "thoughts/Transformers",
    "thoughts/Philosophy-and-Nietzsche",
    "thoughts/Camus",
    "thoughts/atelier-with-friends",
    "thoughts/Attention",
    "thoughts/representations",
  ]

  const largeFiles = vaults!.filter((file) => larges.includes(simplifySlug(file.slug!)))
  const smallFiles = vaults!.filter((file) => smalls.includes(simplifySlug(file.slug!)))

  return h("section", { class: "note-permanent" }, [
    h("h3", { class: "note-subtitle" }, ["persistantes"]),
    h("div", { class: "permanent-grid", style: "position: relative;" }, [
      h(
        "div",
        { class: "large grid-line" },
        largeFiles.map((f) => (
          <a href={resolveRelative(fileData.slug!, f.slug!)} data-list class="perma">
            <div class="title">{f.frontmatter?.title}</div>
            <div class="description">
              {unescapeHTML(
                f.frontmatter?.description ??
                  f.description?.trim() ??
                  i18n(cfg.locale).propertyDefaults.description,
              )}
            </div>
          </a>
        )),
      ),
      h(
        "div",
        { class: "small grid-line" },
        smallFiles.map((f) => (
          <a href={resolveRelative(fileData.slug!, f.slug!)} data-list class="perma">
            <div class="title">{f.frontmatter?.title}</div>
          </a>
        )),
      ),
    ]),
  ])
}

export const Evergreen: QuartzComponent = (props: Props) => {
  const { cfg, allFiles, content } = props
  return (
    <div class="evergreen-content">
      <EvergreenNotes {...props} />
      <AllTags {...props} />
      <article>
        <h3 class="note-subtitle">description</h3>
        {content}
        <p>
          {i18n(cfg.locale).pages.folderContent.itemsUnderFolder({
            count: allFiles.length,
          })}
        </p>
      </article>
    </div>
  )
}

Evergreen.css = style
Evergreen.afterDOMLoaded = script
