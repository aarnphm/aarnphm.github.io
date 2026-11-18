import type { Element, Text } from 'hast'
import type { PhrasingContent } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import type { ReferenceDiagnostic } from './model'
import { escapeHTML } from '../escape'

export type ReferenceLabelNode =
  | { readonly type: 'text' | 'inlineCode'; readonly value: string }
  | { readonly type: 'emphasis' | 'strong'; readonly children: readonly ReferenceLabelNode[] }
  | {
      readonly type: 'link'
      readonly url: string
      readonly children: readonly ReferenceLabelNode[]
    }

export interface ReferenceLabel {
  readonly raw: string
  readonly nodes: readonly ReferenceLabelNode[]
  readonly diagnostics: readonly ReferenceDiagnostic[]
}

function captionLink(url: string): boolean {
  let normalized = ''
  for (const character of url) {
    const code = character.charCodeAt(0)
    if (code > 32 && code !== 127) normalized += character
  }
  const protocol = /^([a-z][a-z\d+.-]*):/i.exec(normalized)
  return !protocol || /^(?:https?|mailto|tel)$/i.test(protocol[1])
}

function phrasing(node: PhrasingContent, links: boolean): ReferenceLabelNode[] {
  if (node.type === 'emphasis' || node.type === 'strong') {
    return [{ type: node.type, children: node.children.flatMap(child => phrasing(child, links)) }]
  }
  if (node.type === 'link' && links && captionLink(node.url))
    return [
      {
        type: 'link',
        url: node.url,
        children: node.children.flatMap(child => phrasing(child, false)),
      },
    ]
  if (node.type === 'text' || node.type === 'inlineCode')
    return [{ type: node.type, value: node.value }]
  if ('children' in node) return node.children.flatMap(child => phrasing(child, links))
  if (node.type === 'image' || node.type === 'imageReference')
    return [{ type: 'text', value: node.alt ?? '' }]
  if (node.type === 'break') return [{ type: 'text', value: '\n' }]
  if ('value' in node && typeof node.value === 'string')
    return [{ type: 'text', value: node.value }]
  return []
}

export function parseReferenceLabel(
  raw: string,
  options: { readonly links?: boolean } = {},
): ReferenceLabel {
  const root = fromMarkdown(raw)
  if (root.children.length === 0) return { raw, nodes: [], diagnostics: [] }
  if (root.children.length !== 1 || root.children[0].type !== 'paragraph') {
    return {
      raw,
      nodes: [{ type: 'text', value: raw }],
      diagnostics: [
        {
          code: 'unsupported-label-block',
          message: 'A reference label must contain inline phrasing only.',
        },
      ],
    }
  }
  return {
    raw,
    nodes: root.children[0].children.flatMap(node => phrasing(node, options.links === true)),
    diagnostics: [],
  }
}

export function labelToHtml(label: ReferenceLabel): string {
  const render = (node: ReferenceLabelNode): string => {
    if ('value' in node)
      return node.type === 'inlineCode'
        ? `<code>${escapeHTML(node.value)}</code>`
        : escapeHTML(node.value)
    if (node.type === 'link')
      return `<a href="${escapeHTML(node.url)}">${node.children.map(render).join('')}</a>`
    const tag = node.type === 'emphasis' ? 'em' : 'strong'
    return `<${tag}>${node.children.map(render).join('')}</${tag}>`
  }
  return label.nodes.map(render).join('')
}

export function labelToText(label: ReferenceLabel): string {
  const render = (node: ReferenceLabelNode): string =>
    'value' in node ? node.value : node.children.map(render).join('')
  return label.nodes.map(render).join('')
}

export function labelToHast(label: ReferenceLabel): (Element | Text)[] {
  const render = (node: ReferenceLabelNode): Element | Text => {
    if ('value' in node)
      return node.type === 'text'
        ? { type: 'text', value: node.value }
        : {
            type: 'element',
            tagName: 'code',
            properties: {},
            children: [{ type: 'text', value: node.value }],
          }
    return {
      type: 'element',
      tagName: node.type === 'link' ? 'a' : node.type === 'emphasis' ? 'em' : 'strong',
      properties: node.type === 'link' ? { href: node.url } : {},
      children: node.children.map(render),
    }
  }
  return label.nodes.map(render)
}
