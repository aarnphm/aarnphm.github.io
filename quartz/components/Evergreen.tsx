import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { getAllSegmentPrefixes, resolveRelative, SimpleSlug, simplifySlug } from "../util/path"
import { h, VNode } from "preact"
import { i18n } from "../i18n"
//@ts-ignore
import script from "./scripts/evergreen.inline"
import style from "./styles/evergreen.scss"
import { QuartzPluginData } from "../plugins/vfile"

type Props = {
  vaults?: QuartzPluginData[]
  content?: VNode
  opts?: EvergreenNotes
} & QuartzComponentProps

export const AllTags: QuartzComponent = ({ cfg, allFiles }: Props) => {
  const tags = [
    ...new Set(
      allFiles.flatMap((data) => data.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes),
    ),
  ].sort((a, b) => a.localeCompare(b))

  return h("section", { class: "note-tags" }, [
    h("h3", { class: "note-title" }, [i18n(cfg.locale).pages.tagContent.tag]),
    h(
      "div",
      { class: "notes-list" },
      tags.map((tag) => h("div", { class: "note-tag", "data-tag": tag }, [tag])),
    ),
  ])
}

interface EvergreenNotes {
  larges: string[]
  smalls: string[]
  tags: string[]
}

const defaultOpts: EvergreenNotes = { larges: [], smalls: [], tags: [] }

const Notes = ((userOpts?: EvergreenNotes) =>
  ({ fileData, vaults }: Props) => {
    const opts = { ...defaultOpts, ...userOpts }
    const largeFiles = vaults!.filter((file) => opts.larges.includes(simplifySlug(file.slug!)))
    const smallFiles = vaults!.filter((file) => opts.smalls.includes(simplifySlug(file.slug!)))

    const tagItemMap: Map<string, QuartzPluginData[]> = new Map()
    for (const tag of opts.tags) {
      tagItemMap.set(
        tag,
        vaults!.filter((file) =>
          (file.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes).includes(tag),
        ),
      )
    }

    return h("section", { class: "note-permanent" }, [
      h("h3", { class: "note-title" }, ["persistantes"]),
      h("div", { class: "permanent-grid", style: "position: relative;" }, [
        h(
          "div",
          { class: "large grid-line" },
          largeFiles.map((f) =>
            h(
              "a",
              { href: resolveRelative(fileData.slug!, f.slug!), "data-list": true, class: "perma" },
              [
                h("div", { class: "title" }, [f.frontmatter?.title]),
                h("div", { class: "description" }, [f.description!]),
              ],
            ),
          ),
        ),
        h(
          "div",
          { class: "mid grid-line" },
          smallFiles.map((f) =>
            h(
              "a",
              { href: resolveRelative(fileData.slug!, f.slug!), "data-list": true, class: "perma" },
              [h("div", { class: "title" }, [f.frontmatter?.title])],
            ),
          ),
        ),
        h(
          "div",
          { class: "small grid-line" },
          Array.from(tagItemMap.entries()).map(([key, pages]) =>
            h(
              "a",
              {
                href: resolveRelative(fileData.slug!, `tags/${key}` as SimpleSlug),
                "data-list": true,
                "data-tag": key,
                class: "perma",
              },
              [
                h("div", { class: "title" }, [key]),
                h("div", { class: "description" }, [
                  pages.length === 1 ? "1 élément" : `${pages.length} éléments`,
                ]),
              ],
            ),
          ),
        ),
      ]),
    ])
  }) satisfies QuartzComponentConstructor

export default ((opts?: EvergreenNotes) => {
  const Evergreen: QuartzComponent = (props: Props) => {
    const Permanent = Notes(opts)
    const { cfg, allFiles, content } = props
    return (
      <div class="evergreen-content">
        <Permanent {...props} />
        <article style={{ marginBottom: 0 }}>
          <h3 class="note-title">description</h3>
          {content}
          <p>
            {i18n(cfg.locale).pages.folderContent.itemsUnderFolder({
              count: allFiles.length,
            })}
          </p>
        </article>
        <AllTags {...props} opts />
      </div>
    )
  }

  Evergreen.css = style
  Evergreen.afterDOMLoaded = script
  return Evergreen
}) satisfies QuartzComponentConstructor
