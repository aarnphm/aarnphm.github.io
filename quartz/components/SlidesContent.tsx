import { ElementContent, Root, Element } from "hast"
import { h } from "hastscript"
import { visit } from "unist-util-visit"
import type { SlideSection } from "../plugins/transformers/slides"
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import { clone } from "../util/clone"
import { htmlToJsx } from "../util/jsx"
import {
  FullSlug,
  joinSegments,
  pathToRoot,
  stripSlashes,
  isAbsoluteURL,
  resolveRelative,
} from "../util/path"
import { transcludeFinal } from "./renderPage"
// @ts-ignore
import slideScript from "./scripts/slides.inline"
import style from "./styles/slides.scss"

export default (() => {
  const SlidesContent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { fileData } = componentData
    const { htmlAst, filePath } = fileData
    const ast = clone(htmlAst) as Root
    const visited = new Set<FullSlug>([fileData.slug!])

    // Apply transclusion for this page variant (no footnote/reference merging on slides)
    const processed = transcludeFinal(ast, componentData, { visited }, { dynalist: false })

    // Re-resolve links so they are correct from <slug>/slides
    const origSlug = fileData.slug as FullSlug
    const slidesSlug = joinSegments(origSlug, "slides") as FullSlug
    const baseForUrl = `https://local/${stripSlashes(origSlug)}.html`

    const rebaseAttr = (val: string): string => {
      if (!val) return val
      if (val.startsWith("#")) return val
      if (val.startsWith("mailto:") || val.startsWith("tel:") || val.startsWith("data:")) return val
      if (val.startsWith("/static")) return val
      if (isAbsoluteURL(val)) return val

      try {
        const u = new URL(val, baseForUrl)
        const absolutePath = u.pathname + (u.hash ?? "")
        return joinSegments(pathToRoot(slidesSlug), stripSlashes(absolutePath))
      } catch {
        return val
      }
    }

    visit(processed, "element", (node: Element) => {
      const props = node.properties ?? {}
      if (props.href) props.href = rebaseAttr(String(props.href))
      if (props.src) props.src = rebaseAttr(String(props.src))
    })

    const toJsx = (nodes: ElementContent[]) => {
      const container = h("div", nodes as ElementContent[])
      return htmlToJsx(filePath!, container)
    }

    const sections = (fileData.slidesIndex ?? []) as SlideSection[]

    return (
      <div class="slides-root">
        <div class="slides-deck" role="list">
          {sections.map((s, idx) => (
            <section
              role="listitem"
              class="slide"
              data-index={idx}
              id={`slide-${idx}`}
              aria-roledescription="slide"
            >
              {idx === 0 && (
                <p>
                  source:{" "}
                  <a
                    href={resolveRelative(slidesSlug, origSlug)}
                    class="internal"
                    data-slug={resolveRelative(slidesSlug, origSlug)}
                    data-no-popover
                  >
                    text
                  </a>
                  ,{" "}
                  <a data-no-popover data-slug="/" href="/">
                    home
                  </a>
                </p>
              )}
              {toJsx(
                ((processed.children as ElementContent[]) || []).slice(s.startIndex, s.endIndex),
              )}
            </section>
          ))}
        </div>
        <nav class="slides-controls" aria-label="slide controls">
          <button class="prev" aria-label="Previous slide">
            ←
          </button>
          <span class="status" aria-live="polite" />
          <button class="next" aria-label="Next slide">
            →
          </button>
        </nav>
      </div>
    )
  }
  SlidesContent.css = style
  SlidesContent.afterDOMLoaded = slideScript
  return SlidesContent
}) satisfies QuartzComponentConstructor
