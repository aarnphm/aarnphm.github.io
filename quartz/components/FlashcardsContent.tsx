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
import flipScript from './scripts/flashcards.inline'
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
    const pageTitle = fileData.frontmatter?.title ?? 'flashcards'
    const sourceHref = resolveRelative(deckSlug, sourceSlug)

    return (
      <div class="flashcards-root">
        <nav class="flashcards-header" aria-label="flashcards">
          <a
            href={sourceHref}
            class="flashcards-back internal"
            data-slug={sourceHref}
            data-no-popover
            aria-label="back to note"
          >
            ←
          </a>
          <p class="flashcards-title">{pageTitle}</p>
          <p class="flashcards-meta">
            {count} {count === 1 ? 'card' : 'cards'}
          </p>
        </nav>
        <div class="flashcards-drill" data-deck={deckSlug} hidden>
          <button type="button" class="flashcards-drill-toggle">
            start drill
          </button>
          <p class="flashcards-drill-status" aria-live="polite" />
          <div class="flashcards-grade" hidden>
            <button type="button" data-grade="1">
              again
            </button>
            <button type="button" data-grade="2">
              hard
            </button>
            <button type="button" data-grade="3">
              good
            </button>
            <button type="button" data-grade="4">
              easy
            </button>
          </div>
        </div>
        <div class="flashcards-deck" role="list">
          {htmlToJsx(filePath!, h('div', cards))}
        </div>
      </div>
    )
  }
  FlashcardsContent.css = style
  FlashcardsContent.afterDOMLoaded = flipScript
  return FlashcardsContent
}) satisfies QuartzComponentConstructor
