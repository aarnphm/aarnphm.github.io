import type { Element, Root as HastRoot } from 'hast'
import type { Code as MdastCode, Root as MdastRoot } from 'mdast'
import { h, s } from 'hastscript'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { svgOptions } from '../../util/svg-options'
import { isRecord } from '../../util/type-guards'
import { rehypeHighlights, type HighlightsOptions } from './rehype-highlights'

function isElement(node: unknown): node is Element {
  return isRecord(node) && node.type === 'element' && typeof node.tagName === 'string'
}

function classNames(node: Element): string[] {
  if (!Array.isArray(node.properties.className)) return []
  return node.properties.className.filter(value => typeof value === 'string')
}

function hasClass(node: Element, className: string): boolean {
  return classNames(node).includes(className)
}

function hasClipboardButton(node: Element): boolean {
  return node.children.some(child => isElement(child) && hasClass(child, 'clipboard-button'))
}

function preCodeElement(node: Element): Element | undefined {
  if (node.tagName !== 'pre') return undefined
  for (const child of node.children) {
    if (isElement(child) && child.tagName === 'code') return child
  }
  return undefined
}

function isRawCodeBlock(node: unknown): node is Element {
  return isElement(node) && preCodeElement(node) !== undefined
}

function addCopyButtons() {
  return (tree: HastRoot) => {
    visit(
      tree,
      node => isRawCodeBlock(node),
      (node, _idx, parent) => {
        if (!isElement(node)) return
        const codeEl = preCodeElement(node)
        if (!codeEl) return
        const hasDisable = codeEl.properties['data-disable-line-number'] !== undefined
        if (hasDisable) {
          node.properties['data-disable-line-number'] = 'true'
          if (isElement(parent)) {
            parent.properties['data-disable-line-number'] = 'true'
          }
        }
        if (hasClipboardButton(node)) return
        node.children = [
          h('button.clipboard-button', { type: 'button', ariaLabel: 'copy source' }, [
            s(
              'svg',
              { ...svgOptions, viewbox: '0 0 16 16', ariaHidden: true, class: 'copy-icon' },
              [s('use', { href: '#github-copy' })],
            ),
            s(
              'svg',
              { ...svgOptions, viewbox: '0 0 16 16', ariaHidden: true, class: 'check-icon' },
              [s('use', { href: '#github-check' })],
            ),
          ]),
          ...node.children,
        ]
      },
    )
  }
}

export const SyntaxHighlighting: QuartzTransformerPlugin<HighlightsOptions> = (opts = {}) => {
  return {
    name: 'SyntaxHighlighting',
    markdownPlugins() {
      return [
        () => {
          return (tree: MdastRoot) => {
            visit(tree, 'code', (node: MdastCode) => {
              const meta = node.meta ?? ''
              const disableLN = /(?:^|\s)disableLineNumber\s*=\s*true(?:\s|$)/i.test(meta)
              if (disableLN) {
                node.data = node.data ?? {}
                node.data.hProperties = {
                  ...node.data.hProperties,
                  'data-disable-line-number': 'true',
                }
                node.meta = meta
                  .replace(/(?:^|\s)disableLineNumber\s*=\s*false(?:\s|$)/gi, ' ')
                  .trim()
              }
            })
          }
        },
      ]
    },
    htmlPlugins() {
      return [[rehypeHighlights, opts], addCopyButtons]
    },
  }
}
