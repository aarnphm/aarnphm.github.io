import type { Element, ElementContent, Root as HastRoot, RootContent } from 'hast'
import type { Code as MdastCode, Root as MdastRoot } from 'mdast'
import type { VFile } from 'vfile'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import rehypePrettyCode, { Options as CodeOptions, Theme as CodeTheme } from 'rehype-pretty-code'
import { Processor } from 'unified'
import { visit } from 'unist-util-visit'
import { svgOptions } from '../../components/svg'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { isRecord } from '../../util/type-guards'

interface Theme extends Record<string, CodeTheme> {
  light: CodeTheme
  dark: CodeTheme
}

interface Options extends CodeOptions {
  theme?: Theme
  keepBackground?: boolean
}

const defaultOptions: Options = {
  theme: { light: 'github-light', dark: 'github-dark' },
  keepBackground: false,
}

const MAX_HIGHLIGHT_CACHE_ENTRIES = 512
const highlightedCodeCache = new Map<string, Element>()

type HastChild = RootContent | ElementContent
type HastParent = { children: HastChild[] }

interface CodeBlockRef {
  parent: HastParent
  index: number
  key: string
}

interface SyntaxCacheState {
  hit: boolean
  keys: string[]
}

declare module 'vfile' {
  interface DataMap {
    syntaxHighlightingCache?: SyntaxCacheState
  }
}

function isElement(node: unknown): node is Element {
  return isRecord(node) && node.type === 'element' && typeof node.tagName === 'string'
}

function isParent(node: unknown): node is HastParent {
  return isRecord(node) && Array.isArray(node.children)
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

function isHighlightedCodeBlock(node: unknown): node is Element {
  if (!isElement(node)) return false
  if (node.tagName === 'figure' && node.properties.dataRehypePrettyCodeFigure !== undefined) {
    return true
  }
  const code = preCodeElement(node)
  return code !== undefined && code.properties.dataLanguage !== undefined
}

function codeOptionsKey(opts: CodeOptions): string {
  return JSON.stringify(opts, (_key, value) =>
    typeof value === 'function' ? value.toString() : value,
  )
}

function codeBlockKey(optsKey: string, node: Element): string {
  return `${optsKey}:${toHtml(node)}`
}

function cloneElement(node: Element): Element | undefined {
  try {
    return structuredClone(node) as Element
  } catch {
    return undefined
  }
}

function setHighlightedCodeCache(key: string, node: Element): void {
  const cloned = cloneElement(node)
  if (!cloned) return
  highlightedCodeCache.set(key, cloned)
  if (highlightedCodeCache.size <= MAX_HIGHLIGHT_CACHE_ENTRIES) return
  const first = highlightedCodeCache.keys().next().value
  if (first) highlightedCodeCache.delete(first)
}

function collectRawCodeBlocks(parent: HastParent, optsKey: string, blocks: CodeBlockRef[]): void {
  for (let index = 0; index < parent.children.length; index += 1) {
    const child = parent.children[index]
    if (isRawCodeBlock(child)) {
      blocks.push({ parent, index, key: codeBlockKey(optsKey, child) })
      continue
    }
    if (isParent(child)) {
      collectRawCodeBlocks(child, optsKey, blocks)
    }
  }
}

function collectHighlightedCodeBlocks(parent: HastParent, blocks: Element[]): void {
  for (const child of parent.children) {
    if (isHighlightedCodeBlock(child)) {
      blocks.push(child)
      continue
    }
    if (isParent(child)) {
      collectHighlightedCodeBlocks(child, blocks)
    }
  }
}

function syntaxCacheLookup(optsKey: string) {
  return (tree: HastRoot, file: VFile) => {
    const blocks: CodeBlockRef[] = []
    collectRawCodeBlocks(tree, optsKey, blocks)

    if (blocks.length === 0) {
      file.data.syntaxHighlightingCache = { hit: true, keys: [] }
      return
    }

    const cachedBlocks = blocks.map(block => highlightedCodeCache.get(block.key))
    const hit = cachedBlocks.every(Boolean)
    file.data.syntaxHighlightingCache = { hit, keys: blocks.map(block => block.key) }
    if (!hit) return

    blocks.forEach((block, index) => {
      const cached = cachedBlocks[index]
      const cloned = cached ? cloneElement(cached) : undefined
      if (cloned) block.parent.children[block.index] = cloned
    })
  }
}

function cachedPrettyCode(opts: CodeOptions) {
  return function attachPrettyCode(this: Processor) {
    const highlight = rehypePrettyCode.call(this, opts)
    return async (tree: HastRoot, file: VFile) => {
      const state = file.data.syntaxHighlightingCache
      if (state?.hit) return

      const result = highlight ? highlight(tree, file, () => undefined) : undefined
      if (result instanceof Promise) {
        const value = await result
        const blocks: Element[] = []
        collectHighlightedCodeBlocks(tree, blocks)

        if (state && state.keys.length === blocks.length) {
          state.keys.forEach((key, index) => setHighlightedCodeCache(key, blocks[index]))
        }

        return value
      }

      const blocks: Element[] = []
      collectHighlightedCodeBlocks(tree, blocks)

      if (state && state.keys.length === blocks.length) {
        state.keys.forEach((key, index) => setHighlightedCodeCache(key, blocks[index]))
      }

      return result
    }
  }
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
          h('span.clipboard-button', { type: 'button', ariaLabel: 'copy source' }, [
            s('svg', { ...svgOptions, viewbox: '0 -8 24 24', class: 'copy-icon' }, [
              s('use', { href: '#github-copy' }),
            ]),
            s('svg', { ...svgOptions, viewbox: '0 -8 24 24', class: 'check-icon' }, [
              s('use', { href: '#github-check' }),
            ]),
          ]),
          ...node.children,
        ]
      },
    )
  }
}

export const SyntaxHighlighting: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts: CodeOptions = { ...defaultOptions, ...userOpts }
  const optsKey = codeOptionsKey(opts)

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
      return [() => syntaxCacheLookup(optsKey), cachedPrettyCode(opts), addCopyButtons]
    },
  }
}
