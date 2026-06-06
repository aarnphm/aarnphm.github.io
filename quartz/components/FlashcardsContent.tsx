import { ElementContent, Root, Element } from 'hast'
import { h } from 'hastscript'
import { visit } from 'unist-util-visit'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { clone } from '../util/clone'
import { flashcardsSlug } from '../util/flashcards-path'
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
import drillScript from './scripts/flashcards.inline'
import style from './styles/flashcards.scss'

export default (() => {
  const FlashcardsContent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { fileData } = componentData
    const { htmlAst, filePath } = fileData
    const ast = clone(htmlAst) as Root
    const visited = new Set<FullSlug>([fileData.slug!])
    const processed = transcludeFinal(ast, componentData, { visited }, { dynalist: false })

    const origSlug = fileData.slug as FullSlug
    const sourceSlug = fileData.flashcards!.sourceSlug
    const deckSlug = flashcardsSlug(sourceSlug)
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
        return joinSegments(pathToRoot(deckSlug), stripSlashes(absolutePath))
      } catch {
        return val
      }
    }

    visit(processed, 'element', (node: Element) => {
      const props = node.properties ?? {}
      if (props.href) props.href = rebaseAttr(String(props.href))
      if (props.src) props.src = rebaseAttr(String(props.src))
    })

    const cards = (processed.children as ElementContent[]) || []
    const count = fileData.flashcards!.cards.length
    const deckTitle = fileData.frontmatter?.title ?? 'flashcards'
    const sourceHref = resolveRelative(deckSlug, sourceSlug)

    return (
      <div class="flashcards-root" data-deck={deckSlug} data-total={count}>
        <header class="flashcards-bar">
          <a
            href={sourceHref}
            class="flashcards-exit internal"
            data-slug={sourceHref}
            data-no-popover
            aria-label="back to note"
          >
            ←
          </a>
          <div
            class="flashcards-progress"
            role="progressbar"
            aria-label="drill progress"
            aria-valuemin={0}
            aria-valuemax={count}
            aria-valuenow={0}
          >
            <div class="flashcards-progress-fill" />
          </div>
          <span class="flashcards-status" aria-live="polite">
            0 / {count}
          </span>
        </header>
        <div class="flashcards-stage">
          <article class="flashcards-card">
            <div class="flashcards-card-head">
              <h1>{deckTitle}</h1>
            </div>
            <div class="flashcards-card-body">
              {htmlToJsx(filePath!, h('div', { class: 'flashcards-deck', role: 'list' }, cards))}
            </div>
          </article>
        </div>
        <nav class="flashcards-controls" aria-label="drill controls">
          <button type="button" class="fc-btn fc-undo" data-action="undo" disabled>
            undo
            <kbd class="fc-key" aria-hidden="true">
              u
            </kbd>
          </button>
          <span class="fc-spacer" />
          <button type="button" class="fc-btn fc-reveal" data-action="reveal">
            reveal
            <kbd class="fc-key" aria-hidden="true">
              space
            </kbd>
          </button>
          <span class="fc-grades" hidden>
            <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="1">
              forgot
              <kbd class="fc-key" aria-hidden="true">
                1
              </kbd>
            </button>
            <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="2">
              hard
              <kbd class="fc-key" aria-hidden="true">
                2
              </kbd>
            </button>
            <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="3">
              good
              <kbd class="fc-key" aria-hidden="true">
                3
              </kbd>
            </button>
            <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="4">
              easy
              <kbd class="fc-key" aria-hidden="true">
                4
              </kbd>
            </button>
          </span>
          <span class="fc-spacer" />
          <button type="button" class="fc-btn fc-end" data-action="end">
            end
            <kbd class="fc-key" aria-hidden="true">
              e
            </kbd>
          </button>
        </nav>
        <section class="flashcards-finished" aria-live="polite" hidden>
          <h1>session complete</h1>
          <p class="flashcards-summary" />
        </section>
      </div>
    )
  }
  FlashcardsContent.css = style
  FlashcardsContent.afterDOMLoaded = drillScript
  return FlashcardsContent
}) satisfies QuartzComponentConstructor
