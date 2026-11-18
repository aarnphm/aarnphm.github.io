import type { Element, Text } from 'hast'
import type { Extension } from 'mdast-util-from-markdown'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toHast } from 'mdast-util-to-hast'
import type { FullSlug } from '../../path'
import { wikilink } from '../../../extensions/micromark-extension-ofm-wikilinks/syntax'
import { resolveRelative } from '../../path'
import { labelToHast, labelToText, parseReferenceLabel } from '../../references/label'
import {
  parseLiteralReference,
  resolveLiteralReference,
  referenceLabelSource,
  type LiteralReferenceResolutionOptions,
} from '../../references/literal'

export function inlineReferenceNodes(
  source: string,
  currentSlug: FullSlug | undefined,
  options?: LiteralReferenceResolutionOptions,
  plainText = false,
): (Element | Text)[] {
  const references: Extension = {
    enter: {
      wikilink() {
        this.buffer()
      },
    },
    exit: {
      wikilink(token) {
        this.resume()
        const raw = this.sliceSerialize(token)
        const link = parseLiteralReference(raw)
        if (link && plainText) {
          this.enter(
            { type: 'text', value: labelToText(parseReferenceLabel(referenceLabelSource(link))) },
            token,
          )
          this.exit(token)
          return
        }
        const route =
          currentSlug && link && !link.embed
            ? resolveLiteralReference(link, currentSlug, options)
            : null
        if (!currentSlug || !link || !route) {
          this.enter({ type: 'text', value: raw }, token)
          this.exit(token)
          return
        }
        this.enter(
          {
            type: 'link',
            url: `${resolveRelative(currentSlug, route.slug)}${route.anchor ?? ''}`,
            children: [],
            data: {
              hProperties: {
                className: ['internal'],
                'data-slug': `${route.slug}${route.anchor ?? ''}`,
              },
              hChildren: labelToHast(
                parseReferenceLabel(referenceLabelSource(link, route.slug, route.implicitLabel)),
              ),
            },
          },
          token,
        )
        this.exit(token)
      },
    },
  }
  const tree = fromMarkdown(source, { extensions: [wikilink()], mdastExtensions: [references] })
  const html = toHast(tree, {
    handlers: {
      html: (_state, node) =>
        node.type === 'html' ? { type: 'text', value: node.value } : undefined,
      link: (state, node) => {
        if (node.type !== 'link') return undefined
        if (plainText) return state.all(node)
        if (!node.data?.hProperties?.['data-slug']) {
          return {
            type: 'text',
            value: source.slice(node.position?.start.offset, node.position?.end.offset),
          }
        }
        return state.applyData(node, {
          type: 'element',
          tagName: 'a',
          properties: { href: node.url },
          children: [],
        })
      },
      image: (_state, node) =>
        node.type === 'image' ? { type: 'text', value: node.alt ?? '' } : undefined,
    },
  })
  if (!html || html.type !== 'root') return []
  return html.children.flatMap(node => {
    if (node.type === 'element') {
      return node.tagName === 'p'
        ? node.children.filter(
            (child): child is Element | Text => child.type === 'element' || child.type === 'text',
          )
        : [node]
    }
    return node.type === 'text' ? [node] : []
  })
}
