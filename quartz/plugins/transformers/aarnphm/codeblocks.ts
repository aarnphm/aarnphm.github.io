import type { Element, Root as HtmlRoot } from 'hast'
import type { Code, Root } from 'mdast'
import type { VFile } from 'vfile'
import { toHtml } from 'hast-util-to-html'
import { SKIP, visit } from 'unist-util-visit'
import { poetryChildren } from '../../../util/poetry'
import { parseSmsHtmlFragment } from '../sms'
import { quoteBlock } from './quotes'

export interface CodeOptions {
  enable: boolean
  poetry: boolean
  quotes: boolean
  sms: boolean
}

function codeHtml(node: Code, options: CodeOptions, language: string): string | undefined {
  const target: Element = {
    type: 'element',
    tagName: 'p',
    properties: { 'data-codeblock': node.lang },
    children: [{ type: 'text', value: node.value }],
  }
  if (node.lang === 'poetry' || node.lang === 'poem') {
    if (!options.poetry) return node.value
    target.tagName = 'pre'
    target.properties.className = ['poetry']
    target.properties['data-language'] = language
    target.children = poetryChildren(node.value)
  } else if (node.lang === 'sms') {
    if (!options.sms) return node.value
    target.properties.className = ['text']
    const fragment = parseSmsHtmlFragment(node.value)
    target.tagName = fragment.wrapperTagName
    target.children = fragment.children
  } else if (node.lang === 'quotes') return node.value
  else return undefined
  return toHtml(target, { allowDangerousHtml: true })
}

export function transformCodeblocks(
  tree: Root,
  file: VFile,
  options: CodeOptions,
  language: string,
): void {
  visit(tree, 'code', (node, index, parent) => {
    if (!parent || index === undefined) return
    if (node.lang === 'quotes' && options.quotes) {
      parent.children.splice(index, 1, quoteBlock(node, String(file.value)))
      return [SKIP, index + 1]
    }
    const value = codeHtml(node, options, language)
    if (value !== undefined)
      parent.children.splice(index, 1, { type: 'html', value, position: node.position })
  })
}

/** Display substitutions leave reference syntax and URL attributes unchanged. */
export function quoteDisplayDashes(tree: HtmlRoot): void {
  visit(tree, 'element', node => {
    if (
      node.tagName !== 'blockquote' ||
      !Array.isArray(node.properties.className) ||
      !node.properties.className.includes('quotes')
    )
      return
    visit(node, descendant => {
      if (descendant.type === 'text') descendant.value = descendant.value.replace(/--/g, '—')
      else if (descendant.type === 'element')
        for (const key of ['alt', 'title']) {
          const value = descendant.properties[key]
          if (typeof value === 'string') descendant.properties[key] = value.replace(/--/g, '—')
        }
    })
    return SKIP
  })
}
