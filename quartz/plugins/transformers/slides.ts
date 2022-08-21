import { QuartzTransformerPlugin } from "../types"
import { Element, ElementContent, Root } from "hast"
import { headingRank } from "hast-util-heading-rank"
import { toString as hastToString } from "hast-util-to-string"

export type SlideSection = {
  type: "intro" | "section"
  id?: string
  title?: string
  level?: number
  startIndex: number
  endIndex: number // exclusive
}

export const Slides: QuartzTransformerPlugin = () => {
  return {
    name: "Slides",
    htmlPlugins() {
      return [
        () => {
          return (tree: Root, file) => {
            // Only generate slides index when explicitly enabled in frontmatter
            if (!file.data.frontmatter?.slides) {
              return
            }
            const children = tree.children as ElementContent[]
            const sections: SlideSection[] = []

            // find first heading index
            let firstHeadingIdx = -1
            for (let i = 0; i < children.length; i++) {
              const el = children[i]
              if (el?.type === "element" && headingRank(el as Element)) {
                firstHeadingIdx = i
                break
              }
            }

            // intro section (content before first heading)
            if (firstHeadingIdx > 0) {
              sections.push({
                type: "intro",
                title: "intro",
                startIndex: 0,
                endIndex: firstHeadingIdx,
              })
            } else if (firstHeadingIdx === -1 && children.length > 0) {
              // No headings at all; treat whole doc as single intro slide
              sections.push({
                type: "intro",
                title: "intro",
                startIndex: 0,
                endIndex: children.length,
              })
            }

            // section for each heading
            for (
              let i = firstHeadingIdx === -1 ? children.length : firstHeadingIdx;
              i < children.length;
              i++
            ) {
              const el = children[i]
              if (!(el?.type === "element" && headingRank(el as Element))) continue

              const startIndex = i
              const level = headingRank(el as Element) as number
              let endIndex = children.length
              for (let j = i + 1; j < children.length; j++) {
                const next = children[j]
                if (next?.type === "element" && headingRank(next as Element)) {
                  const nextLevel = headingRank(next as Element) as number
                  if (nextLevel <= level) {
                    endIndex = j
                    break
                  }
                }
              }

              const headingEl = el as Element
              const id = (headingEl.properties?.id as string) || undefined
              const title = hastToString(headingEl)

              sections.push({
                type: "section",
                id,
                title,
                level,
                startIndex,
                endIndex,
              })

              i = endIndex - 1 // skip to end of section
            }

            file.data.slidesIndex = sections
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    slidesIndex: SlideSection[]
  }
}
