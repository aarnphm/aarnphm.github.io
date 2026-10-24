import {
  initializeTreeSitter,
  parse,
  registerDynamicLanguage,
  type SgRoot,
  type SgNode,
} from '@ast-grep/wasm'
import { Parser } from 'web-tree-sitter'
import type {
  CodePoint,
  CodeQuery,
  CodeRange,
  StructuralResult,
  StructuralSearchOptions,
  StructuralSource,
} from '../util/code-search-types'
import { releaseStructuralTrees } from './code-search-ast-parser'
import { grammarUrl, runtimeUrl } from './code-search-ast-urls'

let initialized: Promise<void> | undefined
const languages = new Map<string, Promise<void>>()

async function initialize(language: string, libraryPath: string): Promise<void> {
  initialized ??= Parser.init({ locateFile: () => runtimeUrl }).then(() => initializeTreeSitter())
  await initialized
  let loading = languages.get(language)
  if (!loading) {
    loading = registerDynamicLanguage({
      [language]: { libraryPath, expandoChar: language === 'python' ? 'µ' : '$' },
    })
    languages.set(language, loading)
  }
  await loading
}

function point(text: string, offset: number): CodePoint {
  const prefix = text.slice(0, offset)
  return { offset, line: prefix.split('\n').length, column: offset - prefix.lastIndexOf('\n') }
}

function rangeOf(node: SgNode, text: string): CodeRange {
  const range = node.range()
  const start = range.start
  const end = range.end
  try {
    return { start: point(text, start.index), end: point(text, end.index) }
  } finally {
    start.free()
    end.free()
    range.free()
  }
}

export async function searchStructural(
  sources: readonly StructuralSource[],
  query: CodeQuery,
  options: StructuralSearchOptions,
): Promise<StructuralResult> {
  const result: StructuralResult = {
    matches: [],
    unsupportedLanguages: [],
    parseErrors: 0,
    scannedSources: 0,
  }
  const unsupported = new Set<string>()
  let sliceStarted = performance.now()
  let reportedScanned = -1
  for (const source of sources) {
    if (options.cancelled()) break
    if (query.language && source.language !== query.language) continue
    const url = grammarUrl(source.language)
    if (!url) {
      unsupported.add(source.language)
    } else {
      await initialize(source.language, url)
      if (options.cancelled()) break
      let document: SgRoot | undefined
      let root: SgNode | undefined
      let matches: SgNode[] = []
      try {
        document = parse(source.language, source.text)
        root = document.root()
        const tree: unknown = document.getInnerTree()
        if (tree && typeof tree === 'object' && 'rootNode' in tree) {
          const node = tree.rootNode
          if (node && typeof node === 'object' && 'hasError' in node && node.hasError)
            result.parseErrors++
        }
        matches = root.findAll({ rule: { pattern: query.pattern } })
        if (matches.length)
          result.matches.push({
            sourceId: source.id,
            ranges: matches.map(node => rangeOf(node, source.text)),
          })
      } finally {
        for (const match of matches) match.free()
        root?.free()
        document?.free()
        releaseStructuralTrees()
      }
    }
    result.scannedSources++
    if (performance.now() - sliceStarted >= 8) {
      options.progress?.(result.scannedSources, sources.length)
      reportedScanned = result.scannedSources
      await new Promise<void>(resolve => setTimeout(resolve, 0))
      sliceStarted = performance.now()
    }
  }
  if (reportedScanned !== result.scannedSources)
    options.progress?.(result.scannedSources, sources.length)
  result.unsupportedLanguages = [...unsupported].sort()
  return result
}
