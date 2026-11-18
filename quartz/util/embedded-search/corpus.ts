import type { Nodes } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { toString } from 'mdast-util-to-string'
import { math } from 'micromark-extension-math'
import { visit } from 'unist-util-visit'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { SearchCorpus, SearchDocument, SearchTask } from './types'
import { isJcastCanvasNode } from '../../plugins/transformers/jcast/types'
import { visitJcast } from '../../plugins/transformers/jcast/visitor'
import { dialectFromMarkdown } from '../../plugins/transformers/ofm/dialect'
import { dialectSyntax } from '../../plugins/transformers/ofm/dialect-syntax'
import { referenceSourcePath } from '../references/context'
import { isSearchExcluded } from '../search-exclude'

const cache = new WeakMap<readonly QuartzPluginData[], SearchCorpus>()

function visible(file: QuartzPluginData): boolean {
  const frontmatter = file.frontmatter
  return (
    frontmatter !== undefined &&
    Boolean(file.slug) &&
    file.protectedPassword === undefined &&
    file.protected !== true &&
    frontmatter?.protected !== true &&
    frontmatter?.noindex !== true &&
    frontmatter?.private !== true &&
    frontmatter?.draft !== true &&
    frontmatter?.publish !== false &&
    !isSearchExcluded(file.slug ?? '') &&
    !isSearchExcluded(referenceSourcePath(file))
  )
}

function bodySource(source: string): string {
  return source.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '')
}

function canvasText(file: QuartzPluginData): string {
  const canvas = file.canvas
  if (!canvas) return ''
  const text: string[] = []
  visitJcast(canvas, 'canvasNode', node => {
    if (!isJcastCanvasNode(node)) return
    const source = node.data?.canvas
    if (source?.type === 'text' && typeof source.text === 'string') text.push(source.text)
  })
  return text.join('\n\n')
}

function sourceDocument(file: QuartzPluginData): SearchDocument | undefined {
  const sourcePath = referenceSourcePath(file)
  if (!/\.(?:md|canvas)$/i.test(sourcePath)) return undefined
  const source = sourcePath.toLowerCase().endsWith('.canvas')
    ? canvasText(file)
    : bodySource(typeof file.rawMarkdownSource === 'string' ? file.rawMarkdownSource : '')
  if (source.length > 2000000) throw new Error(`source exceeds 2,000,000 characters: ${sourcePath}`)
  interface SourceUnits {
    blocks: string[]
    sections: string[]
    tasks: SearchTask[]
  }
  let units: SourceUnits | undefined
  const semantic = (): SourceUnits => {
    if (units) return units
    const tree = fromMarkdown(source, {
      extensions: [dialectSyntax({ comments: true, enableInlineFootnotes: true }), math()],
      mdastExtensions: [dialectFromMarkdown(), mathFromMarkdown()],
    })
    const blocks: string[] = []
    const tasks: SearchTask[] = []
    const headings: number[] = [0]
    function raw(node: Nodes): string {
      const start = node.position?.start.offset
      const end = node.position?.end.offset
      return typeof start === 'number' && typeof end === 'number'
        ? source.slice(start, end)
        : toString(node)
    }
    visit(tree, node => {
      if (node.type === 'heading' && typeof node.position?.start.offset === 'number')
        headings.push(node.position.start.offset)
      if (['paragraph', 'listItem', 'blockquote', 'code', 'table'].includes(node.type))
        blocks.push(raw(node))
      if (node.type === 'listItem') {
        const marker = /^\s*(?:[-+*]|\d+[.)])\s+\[([^\]])\]\s*/.exec(raw(node))
        if (marker)
          tasks.push({ text: raw(node).slice(marker[0].length), checked: marker[1] !== ' ' })
      }
    })
    const starts = [...new Set(headings)].sort((a, b) => a - b)
    const sections = starts.map((start, index) =>
      source.slice(start, starts[index + 1] ?? source.length),
    )
    units = { blocks, sections, tasks }
    return units
  }

  return {
    slug: file.slug ?? '',
    path: sourcePath,
    title: file.frontmatter?.title ?? sourcePath.slice(sourcePath.lastIndexOf('/') + 1),
    text: source,
    tags: (file.frontmatter?.tags ?? []).map(tag => tag.toLowerCase().replace(/^#/, '')),
    get blocks() {
      return semantic().blocks
    },
    get sections() {
      return semantic().sections
    },
    get tasks() {
      return semantic().tasks
    },
    properties: file.authoredProperties ?? {},
  }
}

export function embeddedSearchCorpus(files: readonly QuartzPluginData[]): SearchCorpus {
  const cached = cache.get(files)
  if (cached) return cached
  const documents: SearchDocument[] = []
  let characters = 0
  for (const file of files) {
    if (!visible(file)) continue
    let document: SearchDocument | undefined
    try {
      document = sourceDocument(file)
    } catch (error) {
      return {
        kind: 'error',
        message: error instanceof Error ? error.message : 'source could not be indexed',
      }
    }
    if (!document) continue
    characters += document.text.length
    if (characters > 50000000)
      return { kind: 'error', message: 'search corpus exceeds 50,000,000 source characters' }
    documents.push(document)
    if (documents.length > 10000)
      return { kind: 'error', message: 'search corpus exceeds 10,000 public documents' }
  }
  documents.sort((a, b) => {
    const nameA = a.path
      .slice(a.path.lastIndexOf('/') + 1)
      .normalize('NFC')
      .toLowerCase()
    const nameB = b.path
      .slice(b.path.lastIndexOf('/') + 1)
      .normalize('NFC')
      .toLowerCase()
    return nameA < nameB ? -1 : nameA > nameB ? 1 : a.path < b.path ? -1 : a.path > b.path ? 1 : 0
  })
  const corpus: SearchCorpus = { kind: 'corpus', documents }
  cache.set(files, corpus)
  return corpus
}
