import { h } from 'hastscript'
import { Root, Parent, BlockContent, Blockquote, DefinitionContent, Paragraph, Code } from 'mdast'
import { ReplaceFunction, findAndReplace as mdastFindReplace } from 'mdast-util-find-and-replace'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { toString } from 'mdast-util-to-string'
import { math } from 'micromark-extension-math'
import { PluggableList } from 'unified'
import { SKIP, visit } from 'unist-util-visit'
import type { Options } from '../ofm'
import { remarkSidenote } from '../../../extensions/micromark-extension-ofm-sidenotes'
import {
  remarkWikilink,
  wikilink,
  wikilinkFromMarkdown,
} from '../../../extensions/micromark-extension-ofm-wikilinks'
import {
  canonicalizeCallout,
  formatMathCalloutTitle,
  isMathCallout,
  isStandaloneProofLine,
  italicizeProofLine,
} from '../../../util/callouts'
import { capitalize } from '../../../util/lang'
import { splitCalloutParagraph } from './callout-title'
import { remarkOfmDialect } from './dialect'
import { markImageGridCandidates } from './image-grid'
import { rawHtmlDialect } from './raw-html-dialect'
import { sidenoteBlockContent } from './sidenote-block'
import { remarkTableReferences } from './table'
const arrowMapping: Record<string, string> = {
  '->': '&rarr;',
  '-->': '&rArr;',
  '=>': '&rArr;',
  '==>': '&rArr;',
  '<-': '&larr;',
  '<--': '&lArr;',
  '<=': '&lArr;',
  '<==': '&lArr;',
}

const countIndent = (line: string) => {
  let total = 0
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === ' ') {
      total += 1
      continue
    }
    if (ch === '\t') {
      total += 4
      continue
    }
    break
  }
  return total
}

const stripIndent = (line: string, indent: number) => {
  if (!line || line.trim() === '') return ''
  let remaining = indent
  let i = 0
  while (i < line.length && remaining > 0) {
    const ch = line[i]
    if (ch === ' ') {
      remaining -= 1
      i++
      continue
    }
    if (ch === '\t') {
      remaining -= 4
      i++
      continue
    }
    break
  }
  return line.slice(i)
}

const readIndentedBlock = (lines: string[], startLineIndex: number, baseIndent: number) => {
  let idx = startLineIndex
  while (idx < lines.length && lines[idx].trim() === '') {
    idx++
  }
  if (idx >= lines.length) return undefined
  const firstIndent = countIndent(lines[idx])
  if (firstIndent <= baseIndent) return undefined
  const contentIndent = firstIndent
  const start = idx
  let end = idx
  idx++
  while (idx < lines.length) {
    const line = lines[idx]
    if (line.trim() === '') {
      end = idx
      idx++
      continue
    }
    const indent = countIndent(line)
    if (indent <= baseIndent) break
    end = idx
    idx++
  }
  const raw = lines
    .slice(start, end + 1)
    .map(line => stripIndent(line, contentIndent))
    .join('\n')
  return { raw, startLine: start + 1, endLine: end + 1 }
}

type CalloutContentNode = BlockContent | DefinitionContent

function setNodeClassName(node: CalloutContentNode, className: string): void {
  node.data = {
    ...node.data,
    hProperties: { ...node.data?.hProperties, className: className.split(/\s+/) },
  }
}

function proofCell(className: string, children: CalloutContentNode[]): Blockquote {
  return {
    type: 'blockquote',
    data: { hName: 'div', hProperties: { className: className.split(/\s+/) } },
    children,
  }
}

function extractProofReason(node: CalloutContentNode): string | undefined {
  if (node.type !== 'paragraph') {
    return undefined
  }

  const lastChild = node.children.at(-1)

  if (!lastChild || lastChild.type !== 'text') {
    return undefined
  }

  const match = lastChild.value.match(proofReasonRegex)

  if (!match || match.index === undefined) {
    return undefined
  }

  const reason = match[1].trim()
  lastChild.value = lastChild.value.slice(0, match.index).trimEnd()

  if (lastChild.value.length === 0) {
    node.children.pop()
  }

  return reason
}

function proofReason(reason: string): Blockquote {
  return proofCell('math-proof-reason', [
    { type: 'paragraph', children: [{ type: 'text', value: reason }] },
  ])
}

function proofRow(node: CalloutContentNode): Blockquote {
  const reason = extractProofReason(node)
  const children = reason
    ? [proofCell('math-proof-step', [node]), proofReason(reason)]
    : [proofCell('math-proof-step', [node])]

  return {
    type: 'blockquote',
    data: {
      hName: 'div',
      hProperties: { className: reason ? ['math-proof-row', 'has-reason'] : ['math-proof-row'] },
    },
    children,
  }
}

function formatProofSection(nodes: CalloutContentNode[]): void {
  for (const [index, node] of nodes.entries()) {
    if (node.type !== 'paragraph') {
      continue
    }

    const standaloneProof = isStandaloneProofLine(node)

    if (!italicizeProofLine(node)) {
      continue
    }

    if (!standaloneProof) {
      continue
    }

    setNodeClassName(node, 'math-proof-label')

    const proofNodes = nodes.slice(index + 1)

    if (proofNodes.length === 0) {
      return
    }

    const proofSection: Blockquote = {
      type: 'blockquote',
      data: { hName: 'div', hProperties: { className: ['math-proof'] } },
      children: proofNodes.map(proofRow),
    }

    nodes.splice(index + 1, proofNodes.length, proofSection)
    return
  }
}

// from https://github.com/escwxyz/remark-obsidian-callout/blob/main/src/index.ts
const calloutRegex = new RegExp(/^\[!([\w-]+)\|?(.+?)?\]([+-]?)/)
const proofReasonRegex = /\s+\[([^\]@\n][^\]\n]*)\]\s*$/
// Allow both ::text{hN}:: and bare ::text:: (defaults to h3)
const markerRegex = new RegExp(/::([^:]+?)(?:\{(h[1-7])\})?::/g)
// bare marker without explicit intensity, defaults to h3
const bareMarkerRegex = new RegExp(/::([^:{}]+?)::/g)

const arrowRegex = new RegExp(/(-{1,2}>|={1,2}>|<-{1,2}|<={1,2})/g)
export function createMarkdownPlugins(opts: Options): PluggableList {
  const plugins: PluggableList = [
    [remarkOfmDialect, opts],
    [remarkTableReferences, opts],
  ]

  // regex replacements
  plugins.push(() => {
    return (tree: Root, file) => {
      const replacements: [RegExp, string | ReplaceFunction][] = []

      if (opts.enableMarker) {
        // handle bare ::text:: markers with default intensity h2
        replacements.push([
          bareMarkerRegex,
          (_value: string, ...capture: string[]) => {
            const [text] = capture
            return { type: 'html', value: `<span class="marker marker-h2">${text}</span>` }
          },
        ])

        // handle explicit ::text{hN}:: markers
        replacements.push([
          markerRegex,
          (_value: string, ...capture: string[]) => {
            const [text, intensity] = capture
            return {
              type: 'html',
              value: `<span class="marker marker-${intensity}">${text}</span>`,
            }
          },
        ])
      }

      if (opts.parseArrows) {
        replacements.push([
          arrowRegex,
          (value: string, ..._capture: string[]) => {
            const maybeArrow = arrowMapping[value]
            if (maybeArrow === undefined) return SKIP
            return { type: 'html', value: `<span>${maybeArrow}</span>` }
          },
        ])
      }

      if (opts.enableInHtmlEmbed) {
        visit(tree, 'html', node => {
          for (const [regex, replace] of replacements) {
            if (typeof replace === 'string') {
              node.value = node.value.replace(regex, replace)
            } else {
              node.value = node.value.replace(regex, (substring: string, ...args) => {
                const replaceValue = replace(substring, ...args)
                if (typeof replaceValue === 'string') {
                  return replaceValue
                } else if (
                  typeof replaceValue === 'object' &&
                  replaceValue !== null &&
                  !Array.isArray(replaceValue) &&
                  replaceValue.type === 'html'
                ) {
                  return replaceValue.value
                } else {
                  return substring
                }
              })
            }
          }
          if (opts.highlight || opts.parseTags)
            node.value = rawHtmlDialect(node.value, opts, file.data)
        })
      }
      mdastFindReplace(tree, replacements)
    }
  })

  plugins.push(remarkWikilink)

  // sidenote parser for {{sidenotes...}} syntax
  const sidenoteExtensions = [wikilink(), math()]
  const sidenoteMdastExtensions = [wikilinkFromMarkdown(), mathFromMarkdown()]
  //@ts-ignore
  plugins.push([
    remarkSidenote,
    { micromarkExtensions: sidenoteExtensions, mdastExtensions: sidenoteMdastExtensions },
  ])

  plugins.push(() => (tree: Root, file) => {
    const nodesToRemove: { parent: Parent; index: number }[] = []
    const source = typeof file?.value === 'string' ? file.value : String(file?.value ?? '')
    const lines = source.split(/\r?\n/)

    visit(tree, 'sidenoteDefinition', (node, index, parent) => {
      if (!parent || typeof index !== 'number') return

      const position = node.position
      if (position?.end?.line) {
        const baseLineIndex = position.end.line - 1
        const baseIndent = countIndent(lines[baseLineIndex] ?? '')
        const block = readIndentedBlock(lines, baseLineIndex + 1, baseIndent)
        if (block && block.raw.trim().length > 0) {
          const parsed = fromMarkdown(block.raw, {
            extensions: sidenoteExtensions,
            mdastExtensions: sidenoteMdastExtensions,
          })
          node.children = sidenoteBlockContent(parsed, block.raw, source, block.startLine)
          for (let i = parent.children.length - 1; i > index; i--) {
            const child = parent.children[i]
            const childPos = child?.position
            if (!childPos) continue
            if (childPos.start.line >= block.startLine && childPos.end.line <= block.endLine) {
              parent.children.splice(i, 1)
            }
          }
          return
        }
      }

      let nextIndex = index + 1
      const contentNodes: (BlockContent | DefinitionContent)[] = []

      while (nextIndex < parent.children.length) {
        const next = parent.children[nextIndex]

        if (next.type === 'code' && !next.lang) {
          const codeValue = next.value
          const parsed = fromMarkdown(codeValue, {
            extensions: sidenoteExtensions,
            mdastExtensions: sidenoteMdastExtensions,
          })
          contentNodes.push(
            ...sidenoteBlockContent(parsed, codeValue, source, next.position?.start.line),
          )
          nodesToRemove.push({ parent, index: nextIndex })
          nextIndex++
        } else {
          break
        }
      }

      if (contentNodes.length > 0) {
        node.children = contentNodes
      }
    })

    for (let i = nodesToRemove.length - 1; i >= 0; i--) {
      const { parent, index } = nodesToRemove[i]
      parent.children.splice(index, 1)
    }
  })

  if (opts.callouts) {
    plugins.push(() => (tree: Root) => {
      const mathCalloutCounters = new Map<string, number>()

      visit(tree, 'blockquote', node => {
        if (node.children.length === 0) {
          return
        }

        // find first line and callout content
        const [firstChild, ...calloutContent] = node.children
        if (firstChild.type !== 'paragraph' || firstChild.children[0]?.type !== 'text') {
          return
        }

        const { title, body } = splitCalloutParagraph(firstChild)
        const firstTitle = title[0]
        if (firstTitle?.type !== 'text') return
        const firstLine = firstTitle.value
        const restOfTitle = title.slice(1)
        if (body.length > 0) calloutContent.unshift({ type: 'paragraph', children: body })

        const match = firstLine.match(calloutRegex)
        if (match && match.input) {
          const [calloutDirective, typeString, calloutMetaData, collapseChar] = match
          const calloutType = canonicalizeCallout(typeString.toLowerCase())
          const mathCallout = isMathCallout(calloutType)
          const calloutNumber = mathCallout
            ? (mathCalloutCounters.get(calloutType) ?? 0) + 1
            : undefined
          if (calloutNumber !== undefined) {
            mathCalloutCounters.set(calloutType, calloutNumber)
          }
          const collapse = collapseChar === '+' || collapseChar === '-'
          const defaultState = collapseChar === '-' ? 'collapsed' : 'expanded'
          const titleContent = match.input.slice(calloutDirective.length).trim()
          const useDefaultTitle = titleContent === '' && restOfTitle.length === 0
          const titleTextNode: Paragraph = { type: 'paragraph', children: restOfTitle }
          const visibleTitleContent =
            mathCallout && calloutNumber !== undefined
              ? (formatMathCalloutTitle(
                  calloutType,
                  titleContent,
                  `${titleContent} ${toString(titleTextNode)}`,
                  calloutNumber,
                ) ?? titleContent)
              : useDefaultTitle
                ? capitalize(typeString).replace(/-/g, ' ')
                : titleContent
          const titleNode: Paragraph = {
            type: 'paragraph',
            children: [{ type: 'text', value: visibleTitleContent + ' ' }, ...restOfTitle],
          }
          const titleChildren: CalloutContentNode[] = [
            {
              type: 'blockquote',
              data: { hName: 'div', hProperties: { className: ['callout-title-inner'] } },
              children: [titleNode],
            },
          ]
          if (collapse) {
            titleChildren.push({
              type: 'paragraph',
              data: {
                hName: 'button',
                hProperties: {
                  className: ['callout-fold-toggle'],
                  type: 'button',
                  'aria-label': 'toggle callout',
                  'aria-expanded': String(defaultState !== 'collapsed'),
                },
                hChildren: [h('.fold-callout-icon')],
              },
              children: [],
            })
          }

          const titleContainer: Blockquote = {
            type: 'blockquote',
            data: { hName: 'div', hProperties: { className: ['callout-title'] } },
            children: titleChildren,
          }

          const blockquoteContent: (BlockContent | DefinitionContent)[] = [titleContainer]
          // replace first line of blockquote with title and rest of the paragraph text
          node.children.splice(0, 1, ...blockquoteContent)

          const classNames = ['callout', calloutType]
          if (mathCallout) {
            classNames.push('callout-math')
          }
          if (collapse) {
            classNames.push('is-collapsible')
          }
          if (defaultState === 'collapsed') {
            classNames.push('is-collapsed')
          }

          // add properties to base blockquote
          node.data = {
            hProperties: {
              ...node.data?.hProperties,
              className: classNames,
              'data-callout': calloutType,
              'data-callout-fold': collapse,
              'data-callout-metadata': calloutMetaData,
              ...(calloutNumber === undefined ? {} : { 'data-callout-number': calloutNumber }),
            },
          }

          // Add callout-content class to callout body if it has one.
          if (calloutContent.length > 0) {
            if (mathCallout) {
              formatProofSection(calloutContent)
            }

            const contentData: BlockContent | DefinitionContent = {
              data: { hProperties: { className: ['callout-content'] }, hName: 'div' },
              type: 'blockquote',
              children: [...calloutContent],
            }
            node.children = [node.children[0], contentData]
          }
        }
      })
    })
  }

  if (opts.mermaid) {
    plugins.push(() => {
      return tree => {
        visit(tree, 'code', (node: Code) => {
          if (node.lang === 'mermaid' || node.lang === 'pele') {
            node.data = {
              hProperties: { className: ['mermaid'], 'data-clipboard': toString(node) },
            }
          }
        })
      }
    })
  }

  if (opts.enableImageGrid) {
    plugins.push(() => markImageGridCandidates)
  }

  return plugins
}
