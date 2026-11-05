import type { FootnoteDefinition, Literal, Nodes, Parent, PhrasingContent, Root } from 'mdast'
import type { Extension } from 'mdast-util-from-markdown'
import type { Plugin } from 'unified'
import type { Options } from '../ofm'
import { dialectSyntax } from './dialect-syntax'
import { lowerHighlightColors } from './highlight-colors'
import { lowerTags } from './tags'
import { lowerTaskMarkers } from './task-markers'

interface Tag extends Literal {
  type: 'ofmTag'
  value: string
}

interface Highlight extends Parent {
  type: 'ofmHighlight'
  children: PhrasingContent[]
}

interface InlineFootnote extends Parent {
  type: 'ofmInlineFootnote'
  children: PhrasingContent[]
}

declare module 'mdast' {
  interface RootContentMap {
    ofmTag: Tag
    ofmHighlight: Highlight
    ofmInlineFootnote: InlineFootnote
  }
  interface PhrasingContentMap {
    ofmTag: Tag
    ofmHighlight: Highlight
    ofmInlineFootnote: InlineFootnote
  }
}

function lowerFootnotes(tree: Root) {
  const reserved = new Set<string>()
  function collect(node: Nodes) {
    if (node.type === 'footnoteDefinition' || node.type === 'footnoteReference')
      reserved.add(node.identifier.toLowerCase())
    if ('children' in node) node.children.forEach(collect)
  }
  collect(tree)
  const definitions: FootnoteDefinition[] = []
  let counter = 0
  function lower(node: Nodes): Nodes {
    if ('children' in node) {
      for (let index = 0; index < node.children.length; index++) {
        const child = node.children[index]
        if (child.type === 'ofmInlineFootnote') {
          lower(child)
          let identifier: string
          do {
            identifier = `generated-inline-footnote-${++counter}`
          } while (reserved.has(identifier))
          reserved.add(identifier)
          definitions.push({
            type: 'footnoteDefinition',
            identifier,
            children: [{ type: 'paragraph', children: child.children, position: child.position }],
            position: child.position,
          })
          node.children[index] = { type: 'footnoteReference', identifier, position: child.position }
        } else lower(child)
      }
    }
    return node
  }
  lower(tree)
  tree.children.push(...definitions)
}

export function dialectFromMarkdown(): Extension {
  return {
    canContainEols: ['ofmInlineFootnote', 'ofmHighlight'],
    enter: {
      ofmTag(token) {
        this.enter({ type: 'ofmTag', value: this.sliceSerialize(token).slice(1) }, token)
      },
      ofmHighlight(token) {
        this.enter({ type: 'ofmHighlight', children: [], data: { hName: 'mark' } }, token)
      },
      ofmComment() {
        this.buffer()
      },
      ofmCommentTail(token) {
        this.enter({ type: 'paragraph', children: [] }, token)
      },
      ofmInlineFootnote(token) {
        this.enter({ type: 'ofmInlineFootnote', children: [] }, token)
      },
    },
    exit: {
      ofmTag(token) {
        this.exit(token)
      },
      ofmHighlight(token) {
        this.exit(token)
      },
      ofmComment() {
        this.resume()
      },
      ofmCommentTail(token) {
        this.exit(token)
      },
      ofmInlineFootnote(token) {
        this.exit(token)
      },
    },
    transforms: [
      tree => {
        lowerFootnotes(tree)
        lowerHighlightColors(tree)
      },
    ],
  }
}

export const remarkOfmDialect: Plugin<[Options], Root> = function (options) {
  const data = this.data()
  const parser = this.parser
  let hasComments = true
  if (parser)
    this.parser = (source, file) => {
      hasComments = source.includes('%%')
      return parser(source, file)
    }
  ;(data.micromarkExtensions ??= []).push(dialectSyntax(options, () => hasComments))
  ;(data.fromMarkdownExtensions ??= []).push(dialectFromMarkdown())
  return (tree, file) => {
    if (options.enableCheckbox) lowerTaskMarkers(tree, String(file.value))
    if (options.parseTags) lowerTags(tree, file.data)
  }
}
