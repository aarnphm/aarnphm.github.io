import type { JSX } from 'preact'
import { ElementContent, Root, Element } from 'hast'
import { toString as hastToString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { visit } from 'unist-util-visit'
import type { SlideSection, SlideSubsection } from '../plugins/transformers/slides'
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

// matches RAIL_KEY in scripts/slides.inline.ts
const RAIL_KEY = 'slides-rail-collapsed'

// Each slide is its own page at <note>/slides/<n>. Every page sits at the same
// depth, so relative URLs hold while the script swaps slides in place.
export const slidePageSlug = (noteSlug: FullSlug, idx: number): FullSlug =>
  joinSegments(noteSlug, 'slides', String(idx + 1)) as FullSlug

export interface PreparedSlides {
  sections: SlideSection[]
  body: (idx: number) => JSX.Element
  weights: number[]
}

export interface SlidesPageData {
  deck: PreparedSlides
  active: number
  fragments: string[]
}

export function prepareSlides(componentData: QuartzComponentProps): PreparedSlides {
  const { fileData } = componentData
  const { htmlAst, filePath } = fileData
  const ast = clone(htmlAst) as Root
  const visited = new Set<FullSlug>([fileData.slug!])

  // Apply transclusion for this page variant (no footnote/reference merging on slides)
  const processed = transcludeFinal(ast, componentData, { visited }, { dynalist: false })

  // Re-resolve links so they are correct from <slug>/slides/<n>
  const origSlug = fileData.slug as FullSlug
  const pageSlug = slidePageSlug(origSlug, 0)
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
      return joinSegments(pathToRoot(pageSlug), stripSlashes(absolutePath))
    } catch {
      return val
    }
  }

  visit(processed, 'element', (node: Element) => {
    const props = node.properties ?? {}
    if (props.href) props.href = rebaseAttr(String(props.href))
    if (props.src) props.src = rebaseAttr(String(props.src))
  })

  const sections = (fileData.slidesIndex ?? []) as SlideSection[]
  const blocks = (processed.children as ElementContent[]) || []
  const slice = (idx: number) => blocks.slice(sections[idx].startIndex, sections[idx].endIndex)
  // ruler segments scale with the slide's text so the deck's bulk is visible
  const weights = sections.map((_, idx) => {
    const chars = slice(idx).reduce((sum, node) => sum + hastToString(node as Element).length, 0)
    return Math.max(1, Math.round(chars / 400))
  })
  // a fresh tree per call: the emitter renders each body into a fragment and a page
  const body = (idx: number) => htmlToJsx(filePath!, h('div', slice(idx)))
  return { sections, body, weights }
}

export default (() => {
  const SlidesContent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const origSlug = componentData.fileData.slug as FullSlug
    const page = componentData.slidesPage as SlidesPageData | undefined
    const deck = page?.deck ?? prepareSlides(componentData)
    const active = page?.active ?? 0
    const fragments = page?.fragments ?? []
    const { sections } = deck
    const last = sections.length - 1
    const sourceHref = resolveRelative(slidePageSlug(origSlug, active), origSlug)
    // sibling pages share a directory: slide n is ./n
    const slideHref = (idx: number) => `./${idx + 1}`
    const slideTitle = (section: SlideSection, idx: number) =>
      section.title?.trim() || `slide ${idx + 1}`
    // the rail lists one nesting level: the shallowest headings under the slide heading
    const railSubsections = (section: SlideSection): SlideSubsection[] => {
      const subs = section.subsections ?? []
      if (subs.length === 0) return subs
      const level = Math.min(...subs.map(sub => sub.level))
      return subs.filter(sub => sub.level === level)
    }
    const rulerColumns = deck.weights.map(w => `${w}fr`).join(' ')
    const iconProps: JSX.SVGAttributes<SVGSVGElement> = {
      class: 'slides-icon',
      viewBox: '0 0 24 24',
      width: 16,
      height: 16,
      fill: 'none',
      stroke: 'currentColor',
      strokeWidth: 1.5,
      strokeLinecap: 'square',
      strokeLinejoin: 'miter',
      'aria-hidden': true,
    }
    // past either end the link loses its href and reads as disabled
    const pagerLink = (idx: number): JSX.AnchorHTMLAttributes<HTMLAnchorElement> =>
      idx < 0 || idx > last ? { role: 'link', 'aria-disabled': true } : { href: slideHref(idx) }

    return (
      <div class="slides-root">
        {/* restore a collapsed rail before first paint; the script syncs the toggle */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{localStorage.getItem(${JSON.stringify(RAIL_KEY)})==="1"&&document.currentScript.parentElement.classList.add("is-rail-collapsed")}catch(e){}`,
          }}
        />
        <nav class="slides-toc" aria-label="slides">
          <div class="slides-toc-header">
            <button
              class="slides-toc-toggle"
              aria-label="hide rail"
              aria-expanded="true"
              aria-controls="slides-toc-list"
              title="toggle rail"
            >
              <svg {...iconProps}>
                <path d="M4 7h16" />
                <path d="M4 12h16" />
                <path d="M4 17h16" />
              </svg>
            </button>
            <a
              href={sourceHref}
              class="slides-toc-back internal"
              data-slug={sourceHref}
              data-no-popover
              aria-label="back to note"
              title="back to note"
            >
              <svg {...iconProps}>
                <path d="M4 11 12 4l8 7" />
                <path d="M6 9.5V20h12V9.5" />
                <path d="M10 20v-5.5h4V20" />
              </svg>
            </a>
          </div>
          <div class="slides-toc-list-scroll" id="slides-toc-list">
            <ol class="slides-toc-list">
              {sections.map((s, idx) => {
                const subs = railSubsections(s)
                return (
                  <li
                    class={idx === active ? 'slides-toc-entry is-active' : 'slides-toc-entry'}
                    data-slide={idx}
                  >
                    <a
                      href={slideHref(idx)}
                      class={[
                        'slides-toc-item',
                        idx === active && 'is-active',
                        idx <= active && 'is-complete',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      data-slide-target={idx}
                      data-no-popover
                      data-router-ignore
                      aria-current={idx === active ? 'step' : undefined}
                    >
                      <span class="slides-toc-label">{slideTitle(s, idx)}</span>
                    </a>
                    {subs.length > 0 && (
                      <ol
                        class="slides-toc-sublist"
                        aria-label={`sections of ${slideTitle(s, idx)}`}
                      >
                        {subs.map((sub, k) => (
                          <li class="slides-toc-subentry" style={`--slides-toc-sub-index: ${k}`}>
                            <a
                              href={`${slideHref(idx)}#${sub.id}`}
                              class="slides-toc-subitem"
                              data-slide-target={idx}
                              data-heading-id={sub.id}
                              data-no-popover
                              data-router-ignore
                            >
                              <span class="slides-toc-label">{sub.title}</span>
                            </a>
                          </li>
                        ))}
                      </ol>
                    )}
                  </li>
                )
              })}
            </ol>
          </div>
        </nav>
        <div class="slides-deck" role="list">
          {sections.map((s, idx) => (
            <section
              role="listitem"
              class={idx === active ? 'slide active' : 'slide'}
              data-index={idx}
              data-title={slideTitle(s, idx)}
              data-src={fragments[idx]}
              data-loaded={idx === active ? '' : undefined}
              id={`slide-${idx}`}
              aria-hidden={idx === active ? 'false' : 'true'}
              aria-roledescription="slide"
            >
              {/* other slides arrive as fragments when the script turns to them */}
              <div class="slide-body">{idx === active ? deck.body(idx) : null}</div>
            </section>
          ))}
        </div>
        <nav class="slides-controls" aria-label="slide controls">
          <div
            class="slides-progress"
            role="progressbar"
            aria-label="slides progress"
            aria-valuemin={0}
            aria-valuemax={sections.length}
            aria-valuenow={active + 1}
            style={`grid-template-columns: ${rulerColumns}`}
          >
            {sections.map((_, idx) => (
              <span
                class={
                  idx <= active ? 'slides-progress-segment is-complete' : 'slides-progress-segment'
                }
              />
            ))}
          </div>
          <span class="status" aria-live="polite">
            {active + 1} / {sections.length}
          </span>
          <span class="slides-timer" hidden />
          <div class="slides-modes">
            <button
              class="overview"
              aria-label="overview"
              aria-pressed="false"
              aria-keyshortcuts="o"
              title="overview (o)"
            >
              <svg {...iconProps}>
                <rect x="4" y="4" width="6.5" height="6.5" />
                <rect x="13.5" y="4" width="6.5" height="6.5" />
                <rect x="4" y="13.5" width="6.5" height="6.5" />
                <rect x="13.5" y="13.5" width="6.5" height="6.5" />
              </svg>
            </button>
            <button
              class="present"
              aria-label="present"
              aria-pressed="false"
              aria-keyshortcuts="p"
              title="present (p)"
              hidden
            >
              <svg {...iconProps}>
                <path d="M4 9.5V4h5.5" />
                <path d="M14.5 4H20v5.5" />
                <path d="M20 14.5V20h-5.5" />
                <path d="M9.5 20H4v-5.5" />
              </svg>
            </button>
          </div>
          <div class="slides-pager">
            {/* links, so the pager works before the script and opens in a new tab */}
            <a
              class="prev"
              data-router-ignore
              data-no-popover
              {...pagerLink(active - 1)}
              aria-label="previous slide"
              aria-keyshortcuts="ArrowLeft"
              title="previous slide (←)"
            >
              <svg {...iconProps}>
                <path d="M19 12H5" />
                <path d="m11.5 5.5-6.5 6.5 6.5 6.5" />
              </svg>
            </a>
            <a
              class="next"
              data-router-ignore
              data-no-popover
              {...pagerLink(active + 1)}
              aria-label="next slide"
              aria-keyshortcuts="ArrowRight"
              title="next slide (→)"
            >
              <svg {...iconProps}>
                <path d="M5 12h14" />
                <path d="m12.5 5.5 6.5 6.5-6.5 6.5" />
              </svg>
            </a>
          </div>
        </nav>
      </div>
    )
  }
  SlidesContent.css = style
  SlidesContent.afterDOMLoaded = slideScript
  return SlidesContent
}) satisfies QuartzComponentConstructor
