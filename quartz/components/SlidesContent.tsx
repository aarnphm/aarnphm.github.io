import { ElementContent, Root, Element } from 'hast'
import { h } from 'hastscript'
import { visit } from 'unist-util-visit'
import type { SlideSection } from '../plugins/transformers/slides'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { clone } from '../util/clone'
import { htmlToJsx } from '../util/jsx'
import {
  FullSlug,
  joinSegments,
  pathToRoot,
  stripSlashes,
  isAbsoluteURL,
  resolveRelative,
} from '../util/path'
import { transcludeFinal } from './renderPage'
// @ts-ignore
import slideScript from './scripts/slides.inline'
import style from './styles/slides.scss'

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
    const slidesSlug = joinSegments(origSlug, 'slides') as FullSlug
    const baseForUrl = `https://local/${stripSlashes(origSlug)}.html`
    const allowedAbsoluteProtocols = new Set(['http:', 'https:', 'mailto:', 'tel:', 'data:'])
    const isAllowedAbsoluteAttr = (value: string): boolean => {
      try {
        return allowedAbsoluteProtocols.has(new URL(value).protocol.toLowerCase())
      } catch {
        return false
      }
    }

    const rebaseAttr = (val: string): string => {
      if (!val) return val
      if (val.startsWith('#')) return val
      if (val.startsWith('/static')) return val
      if (isAbsoluteURL(val)) return isAllowedAbsoluteAttr(val) ? val : ''

      try {
        const u = new URL(val, baseForUrl)
        const absolutePath = u.pathname + (u.hash ?? '')
        return joinSegments(pathToRoot(slidesSlug), stripSlashes(absolutePath))
      } catch {
        return val
      }
    }

    visit(processed, 'element', (node: Element) => {
      const props = node.properties ?? {}
      if (props.href) props.href = rebaseAttr(String(props.href))
      if (props.src) props.src = rebaseAttr(String(props.src))
    })

    const toJsx = (nodes: ElementContent[]) => {
      const container = h('div', nodes as ElementContent[])
      return htmlToJsx(filePath!, container)
    }

    const sections = (fileData.slidesIndex ?? []) as SlideSection[]
    const pageTitle = fileData.frontmatter?.title ?? 'slides'
    const minutes = Math.ceil(fileData.readingTime?.minutes ?? 0)
    const meta = minutes > 0 ? `${minutes} min read` : `${sections.length} slides`
    const sourceHref = resolveRelative(slidesSlug, origSlug)
    const slideTitle = (section: SlideSection, idx: number) =>
      section.title?.trim() || `slide ${idx + 1}`

    return (
      <div class="slides-root">
        <nav class="slides-toc" aria-labelledby="slides-toc-title">
          <div class="slides-toc-header">
            <a
              href={sourceHref}
              class="slides-toc-back internal"
              data-slug={sourceHref}
              data-no-popover
              aria-label="Exit slides mode"
            >
              ←
            </a>
            <p class="slides-toc-title" id="slides-toc-title">
              {pageTitle}
            </p>
            <p class="slides-toc-meta">{meta}</p>
          </div>
          <div class="slides-toc-list-scroll">
            <ol class="slides-toc-list" style="--slides-toc-progress: 0">
              {sections.map((s, idx) => (
                <li class="slides-toc-entry">
                  <a
                    href={`#slide-${idx}`}
                    class={idx === 0 ? 'slides-toc-item is-active is-complete' : 'slides-toc-item'}
                    data-slide-target={idx}
                    data-no-popover
                    aria-current={idx === 0 ? 'step' : undefined}
                  >
                    {slideTitle(s, idx)}
                  </a>
                </li>
              ))}
            </ol>
          </div>
        </nav>
        <div class="slides-deck" role="list">
          {sections.map((s, idx) => (
            <section
              role="listitem"
              class={idx === 0 ? 'slide active' : 'slide'}
              data-index={idx}
              id={`slide-${idx}`}
              aria-hidden={idx === 0 ? 'false' : 'true'}
              aria-roledescription="slide"
            >
              {idx === 0 && (
                <p>
                  source:{' '}
                  <a href={sourceHref} class="internal" data-slug={sourceHref} data-no-popover>
                    text
                  </a>
                  ,{' '}
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
          <div
            class="slides-progress"
            role="progressbar"
            aria-label="Slides progress"
            aria-valuemin={0}
            aria-valuemax={sections.length}
            aria-valuenow={1}
          >
            <div
              class="slides-progress-bar"
              style={`width: ${sections.length ? (1 / sections.length) * 100 : 0}%`}
            />
          </div>
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
