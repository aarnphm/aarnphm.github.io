import type { ElementContent, RootContent } from 'hast'
import { Root, Element } from 'hast'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toHtml } from 'hast-util-to-html'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import isAbsoluteUrl from 'is-absolute-url'
import { Root as MdastRoot, Code } from 'mdast'
import path from 'path'
import { SKIP, visit } from 'unist-util-visit'
// @ts-ignore
import script from '../../components/scripts/telescopic.inline'
import content from '../../components/styles/telescopic.inline.scss'
// A small subsets of https://github.com/jackyzha0/telescopic-text
// turned into remark plugin for Quartz
//
// Usage:
// ````
// ```telescopic id="unique-block"
// * reading
//   * reading a lot of Nietzsche,
//   * hosting functions,
//     * go on longs walks,
//     * building [[thoughts/work|open-source project]],
// ```
// ````
//
// NOTE: For this, we only a small subset of wikilinks, and it should always be absolute
//
// Some deviation: separator=" ", shouldExpandOnMouseOver=false
import { QuartzTransformerPlugin } from '../../types/plugin'
import { simplifySlug, splitAnchor, stripSlashes, transformLink } from '../../util/path'
import { type Content, telescopeHast, telescopicContent } from '../../util/telescopic'
import { extractWikilinksWithPositions } from '../../util/wikilinks'

interface Config {
  /**
   * Character used to separate entries on the same level. Defaults to a single space (" ")
   */
  separator?: string
  /**
   * If true, allows sections to expand automatically on mouse over rather than requiring a click. Defaults to false.
   */
  shouldExpandOnMouseOver?: boolean
}

function applyInlineFormatting(value: string): string {
  if (!value) {
    return value
  }

  let formatted = value

  // Normalize hard and soft line breaks
  formatted = formatted.replace(/\\n/g, '<br />')
  formatted = formatted.replace(/\\(?=\s|$)/g, '<br />')
  formatted = formatted.replace(/\n\s*\n\s*\n/g, '<hr />')
  formatted = formatted.replace(/\n\s*\n/g, '<hr />')
  formatted = formatted.replace(/\r?\n/g, '<br />')
  formatted = formatted.replace(/<br\s*\/?>/gi, '<br />')
  formatted = formatted.replace(/&softbreak;/gi, '<wbr />')
  formatted = formatted.replace(/<softbreak\s*\/?>/gi, '<wbr />')

  // Bold / strong emphasis (**text** or __text__)
  formatted = formatted.replace(/(?<!\\)\*\*([\s\S]+?)\*\*/g, '<strong>$1</strong>')
  formatted = formatted.replace(/(?<!\\)__([\s\S]+?)__/g, '<strong>$1</strong>')

  // Italic emphasis (*text* or _text_)
  formatted = formatted.replace(/(?<!\\)\*(?!\*)([^*\n]+?)\*(?!\*)/g, '<em>$1</em>')
  formatted = formatted.replace(/(?<!\\)_(?!_)([^_\n]+?)_(?!_)/g, '<em>$1</em>')

  // Strikethrough
  formatted = formatted.replace(/(?<!\\)~~([\s\S]+?)~~/g, '<del>$1</del>')

  // Restore escaped characters that should remain literal
  formatted = formatted.replace(/\\\*/g, '*')
  formatted = formatted.replace(/\\_/g, '_')
  formatted = formatted.replace(/\\~/g, '~')
  formatted = formatted.replace(/\\\\/g, '\\')

  return formatted
}

function inlineFragmentToNodes(value: string): RootContent[] {
  if (!value) {
    return []
  }

  const formatted = applyInlineFormatting(value)
  const fragment = fromHtmlIsomorphic(formatted, { fragment: true }) as Root
  const children = (fragment.children ?? []) as RootContent[]
  return children
}

const contentToHast = (content: Content, opts: Config) =>
  telescopeHast(
    content,
    value => inlineFragmentToNodes(value) as ElementContent[],
    opts.shouldExpandOnMouseOver ? 'hover' : 'click',
  )

const mdToContent = (mdContent: string, separator: string = ' ') =>
  telescopicContent(mdContent, separator, line => console.log(`Invalid line found! ${line}`))

const defaultOpts: Config = { separator: ' ', shouldExpandOnMouseOver: false }

const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g

export const TelescopicText: QuartzTransformerPlugin<Partial<Config>> = userOpts => {
  const opts = { ...defaultOpts, ...userOpts }

  return {
    name: 'TelescopicText',
    markdownPlugins() {
      return [
        () => (tree: MdastRoot, file) => {
          visit(tree, 'code', (node: Code, index, parent) => {
            let { lang, meta } = node
            if (lang === 'telescopic' && meta) {
              // Parse the meta string to extract ID
              const idMatch = meta!.match(/id=["']?([^"'\s]+)["']?/)
              if (!idMatch) {
                console.error(
                  `Missing id for targeted telescopic node ${node.position}, skipping instead (${file.data.filePath!})`,
                )
                return SKIP
              }
              node.data = node.data || {}
              node.data.hProperties = { id: idMatch[1] + '-container' }
              const mark = meta.match(/(?:^|\s)marks=["']?([\w-]+)["']?(?=\s|$)/)?.[1]
              if (mark && parent && index !== undefined) {
                // Let the Obsidian parser register the final rendered container.
                parent.children.splice(index + 1, 0, {
                  type: 'paragraph',
                  children: [{ type: 'text', value: `^${mark}` }],
                })
                return index + 2
              }
            }
          })
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => (tree: Root, file) => {
          const curSlug = simplifySlug(file.data.slug!)

          const resolveWikilink = (
            target: string,
            anchor: string | undefined,
            alias: string | undefined,
          ): { dest: string; alias: string } => {
            const fp = target.trim()
            const resolvedAnchor = anchor?.trim() ?? ''
            const resolvedAlias = alias?.trim()

            let dest = fp + resolvedAnchor
            const ext: string = path.extname(dest).toLowerCase()
            if (isAbsoluteUrl(dest, { httpOnly: false })) return { dest, alias: dest }

            if (!ext.includes('pdf')) {
              dest = transformLink(file.data.slug!, dest, {
                allSlugs: ctx.allSlugs,
                strategy: 'absolute',
              })
            }

            // url.resolve is considered legacy
            // WHATWG equivalent https://nodejs.dev/en/api/v18/url/#urlresolvefrom-to
            const url = new URL(dest, 'https://base.com/' + stripSlashes(curSlug, true))
            const canonicalDest = url.pathname
            let [destCanonical, _destAnchor] = splitAnchor(canonicalDest)
            if (destCanonical.endsWith('/')) {
              destCanonical += 'index'
            }
            return { dest, alias: resolvedAlias ?? path.basename(fp) }
          }

          const checkParsedCodeblock = ({ tagName, children }: Element): boolean => {
            if (tagName !== 'pre' || !Array.isArray(children) || children.length === 0) {
              return false
            }

            const [code] = children as Element[]
            const { properties, tagName: codeTagName } = code
            return (
              codeTagName === 'code' &&
              Boolean(properties.className) &&
              (properties.className as string[]).includes('language-telescopic')
            )
          }

          visit(
            tree,
            node => checkParsedCodeblock(node as Element),
            (node, index, parent) => {
              const code = (node as Element).children[0] as Element

              let codeText = toString(code)
              const ranges = extractWikilinksWithPositions(codeText)
              if (ranges.length > 0) {
                let rewritten = ''
                let lastIndex = 0
                for (const range of ranges) {
                  if (range.start > lastIndex) {
                    rewritten += codeText.slice(lastIndex, range.start)
                  }

                  const parsed = range.wikilink
                  const { dest, alias } = resolveWikilink(
                    parsed.target,
                    parsed.anchor,
                    parsed.alias,
                  )
                  rewritten += toHtml(h('a', { href: dest }, { type: 'text', value: alias }))
                  lastIndex = range.end
                }
                if (lastIndex < codeText.length) {
                  rewritten += codeText.slice(lastIndex)
                }
                codeText = rewritten
              }

              codeText = codeText.replace(linkRegex, (_match, value, href) => {
                return toHtml(h('a', { href }, { type: 'text', value }))
              })
              code.children = [{ type: 'text', value: codeText }]

              const content = mdToContent(codeText, opts.separator)
              parent!.children.splice(
                index!,
                1,
                h(
                  'div.telescopic-container',
                  { id: code.properties.id },
                  contentToHast(content, opts),
                ),
              )
            },
          )
        },
      ]
    },
    externalResources() {
      return {
        js: [{ script, contentType: 'inline', loadTime: 'afterDOMReady' }],
        css: [{ content, spaPreserve: true, inline: true }],
      }
    },
  }
}
