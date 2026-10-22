import type { Completion } from '@codemirror/autocomplete'
import type { Text } from '@codemirror/state'
import type { LeanCatalogEntry } from './catalog-entry'
import { LEAN_STATIC_CATALOG } from './static-catalog'

export type IndexedCatalogEntry = {
  readonly entry: LeanCatalogEntry
  readonly label: string
  readonly labelLower: string
  readonly hasDot: boolean
  readonly namespacePrefix: string
  readonly memberLower: string
  readonly isCommand: boolean
  readonly isLocal: boolean
  readonly boost45: Completion
  readonly boost40: Completion
  readonly boost30: Completion
  readonly boost25: Completion
  readonly boost18: Completion
  readonly boost10: Completion
  readonly boost5: Completion
}

function createCompletionWithBoost(entry: LeanCatalogEntry, boost: number): Completion {
  return {
    label: entry.label,
    type: entry.type,
    detail: entry.detail,
    info: entry.documentation,
    boost,
  }
}

function indexCatalogEntry(entry: LeanCatalogEntry): IndexedCatalogEntry {
  const label = entry.label
  const labelLower = label.toLowerCase()
  const dotIndex = label.lastIndexOf('.')
  const hasDot = dotIndex !== -1
  const namespacePrefix = hasDot ? label.slice(0, dotIndex + 1) : ''
  const memberLower = hasDot ? labelLower.slice(dotIndex + 1) : labelLower
  const isCommand = label.charCodeAt(0) === 35
  const isLocal = entry.category === 'local'
  return {
    entry,
    label,
    labelLower,
    hasDot,
    namespacePrefix,
    memberLower,
    isCommand,
    isLocal,
    boost45: createCompletionWithBoost(entry, 45),
    boost40: createCompletionWithBoost(entry, 40),
    boost30: createCompletionWithBoost(entry, 30),
    boost25: createCompletionWithBoost(entry, 25),
    boost18: createCompletionWithBoost(entry, 18),
    boost10: createCompletionWithBoost(entry, 10),
    boost5: createCompletionWithBoost(entry, 5),
  }
}

export const LEAN_CATALOG_BY_LABEL = new Map<string, LeanCatalogEntry>()
export const STATIC_COMMANDS: IndexedCatalogEntry[] = []
export const STATIC_NON_COMMANDS: IndexedCatalogEntry[] = []
export const STATIC_BY_NAMESPACE = new Map<string, IndexedCatalogEntry[]>()

for (const entry of LEAN_STATIC_CATALOG) {
  LEAN_CATALOG_BY_LABEL.set(entry.label, entry)
  const indexed = indexCatalogEntry(entry)
  if (indexed.isCommand) {
    STATIC_COMMANDS.push(indexed)
  } else {
    STATIC_NON_COMMANDS.push(indexed)
    if (indexed.hasDot) {
      let bucket = STATIC_BY_NAMESPACE.get(indexed.namespacePrefix)
      if (!bucket) {
        bucket = []
        STATIC_BY_NAMESPACE.set(indexed.namespacePrefix, bucket)
      }
      bucket.push(indexed)
    }
  }
}

const RESERVED_SKIP_NAMES = new Set([
  'by',
  'in',
  'with',
  'where',
  'do',
  'fun',
  'match',
  'if',
  'then',
  'else',
  'this',
  'rfl',
  'sorry',
  'admit',
])

const DECL_START_WORDS = new Set([
  'theorem',
  'lemma',
  'def',
  'abbrev',
  'structure',
  'class',
  'inductive',
  'axiom',
  'opaque',
  'noncomputable',
  'private',
  'protected',
  'scoped',
  'local',
  'partial',
  'unsafe',
])

const VALID_SYMBOL_NAME_RE =
  /^[A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_.'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*$/
const DECL_LINE_RE =
  /^(?:(?:noncomputable|private|protected|scoped|local|partial|unsafe)\s+)*(theorem|lemma|def|abbrev|structure|class|inductive|axiom|opaque)\s+([A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_.'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*)(.*)$/
const BINDER_GROUP_RE = /[(⦃{]\s*([^():⦃⦄{}]+?)\s*:\s*([^()⦃⦄{}]+?)[)⦄}]/g
const HAVE_LET_LINE_RE =
  /\b(have|let)\s+([A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*)\s*(?::\s*([^:=]+?))?\s*:=/
const OBTAIN_RCASES_LINE_RE = /\b(?:obtain|rcases)\b[^⟨]*⟨([^⟩]+)⟩/g
const INTRO_LINE_RE =
  /\b(?:intro|intros)\s+([A-Za-z0-9_'!?\s\u00c0-\u024f\u0370-\u03ff\u2080-\u2089]+)/

type LeanDocumentSymbolCache = {
  readonly entries: readonly LeanCatalogEntry[]
  readonly indexed: readonly IndexedCatalogEntry[]
  readonly byLabel: ReadonlyMap<string, LeanCatalogEntry>
  readonly completionCache: Map<string, readonly Completion[]>
}

export const EMPTY_INDEXED_ENTRIES: readonly IndexedCatalogEntry[] = []
const docSymbolCacheByText = new WeakMap<Text, LeanDocumentSymbolCache>()
const docSymbolCacheBySource = new Map<string, LeanDocumentSymbolCache>()
const MAX_DOC_SYMBOL_CACHE = 64

function scanSourceIntoSymbols(
  source: string,
  addSymbol: (label: string, type: string, detail: string, documentation: string) => void,
): void {
  let lineStart = 0
  const len = source.length
  while (lineStart < len) {
    const nlIdx = source.indexOf('\n', lineStart)
    const lineEnd = nlIdx === -1 ? len : nlIdx
    const rawLine = source.slice(lineStart, lineEnd)
    lineStart = nlIdx === -1 ? len : nlIdx + 1

    const commentIdx = rawLine.indexOf('--')
    const line = (commentIdx !== -1 ? rawLine.slice(0, commentIdx) : rawLine).trim()
    if (!line) continue

    const firstSpace = line.indexOf(' ')
    if (firstSpace !== -1 && DECL_START_WORDS.has(line.slice(0, firstSpace))) {
      const declMatch = line.match(DECL_LINE_RE)
      if (declMatch) {
        const kind = declMatch[1]
        const name = declMatch[2]
        const rest =
          declMatch[3]
            ?.replace(/:=\s*by\b.*$/, '')
            .replace(/:=.*$/, '')
            .trim() ?? ''
        const isType = kind === 'structure' || kind === 'class' || kind === 'inductive'
        addSymbol(
          name,
          isType ? 'type' : 'function',
          rest ? `${kind} ${name} ${rest}` : `${kind} ${name}`,
          `Document-local ${kind} \`${name}\`.`,
        )

        if (rest.indexOf(':') !== -1) {
          for (const binder of rest.matchAll(BINDER_GROUP_RE)) {
            const namesPart = binder[1].trim()
            const typePart = binder[2].trim()
            for (const param of namesPart.split(/\s+/)) {
              addSymbol(
                param,
                'variable',
                `${param} : ${typePart}`,
                `Parameter \`${param} : ${typePart}\` of \`${name}\`.`,
              )
            }
          }
        }
      }
    }

    if (
      line.indexOf(':=') !== -1 &&
      (line.indexOf('have ') !== -1 || line.indexOf('let ') !== -1)
    ) {
      const haveLetMatch = line.match(HAVE_LET_LINE_RE)
      if (haveLetMatch) {
        const kind = haveLetMatch[1]
        const name = haveLetMatch[2]
        const typeStr = haveLetMatch[3]?.trim()
        addSymbol(
          name,
          'variable',
          typeStr ? `${name} : ${typeStr}` : `${kind} ${name}`,
          `Local ${kind === 'have' ? 'hypothesis' : 'binding'} \`${name}\`${
            typeStr ? ` : \`${typeStr}\`` : ''
          }.`,
        )
      }
    }

    if (
      line.indexOf('⟨') !== -1 &&
      (line.indexOf('obtain') !== -1 || line.indexOf('rcases') !== -1)
    ) {
      for (const tupleMatch of line.matchAll(OBTAIN_RCASES_LINE_RE)) {
        for (const part of tupleMatch[1].split(/[,\s|⟨⟩]+/)) {
          addSymbol(
            part,
            'variable',
            `hypothesis ${part}`,
            `Local destructured witness or hypothesis \`${part}\`.`,
          )
        }
      }
    }

    if (line.indexOf('intro') !== -1) {
      const introMatch = line.match(INTRO_LINE_RE)
      if (introMatch) {
        for (const ident of introMatch[1].split(/\s+/)) {
          addSymbol(ident, 'variable', `hypothesis ${ident}`, `Local hypothesis \`${ident}\`.`)
        }
      }
    }
  }
}

export function extractLeanDocumentSymbols(sources: readonly string[]): LeanCatalogEntry[] {
  if (sources.length === 1) {
    return [...getDocumentSymbolsForSource(sources[0]!).entries]
  }
  const seen = new Set<string>()
  const out: LeanCatalogEntry[] = []

  const addSymbol = (label: string, type: string, detail: string, documentation: string) => {
    const clean = label.trim()
    if (
      !clean ||
      clean === '_' ||
      RESERVED_SKIP_NAMES.has(clean) ||
      LEAN_CATALOG_BY_LABEL.has(clean) ||
      seen.has(clean) ||
      !VALID_SYMBOL_NAME_RE.test(clean)
    ) {
      return
    }
    seen.add(clean)
    out.push({ label: clean, category: 'local', type, detail, documentation })
  }

  for (const source of sources) {
    scanSourceIntoSymbols(source, addSymbol)
  }

  return out
}

export function getDocumentSymbolsForSource(source: string): LeanDocumentSymbolCache {
  const existing = docSymbolCacheBySource.get(source)
  if (existing) return existing

  const byLabel = new Map<string, LeanCatalogEntry>()
  const entries: LeanCatalogEntry[] = []
  const indexed: IndexedCatalogEntry[] = []

  scanSourceIntoSymbols(source, (label, type, detail, documentation) => {
    const clean = label.trim()
    if (
      !clean ||
      clean === '_' ||
      RESERVED_SKIP_NAMES.has(clean) ||
      LEAN_CATALOG_BY_LABEL.has(clean) ||
      byLabel.has(clean) ||
      !VALID_SYMBOL_NAME_RE.test(clean)
    ) {
      return
    }
    const entry: LeanCatalogEntry = { label: clean, category: 'local', type, detail, documentation }
    byLabel.set(clean, entry)
    entries.push(entry)
    indexed.push(indexCatalogEntry(entry))
  })

  const record: LeanDocumentSymbolCache = {
    entries,
    indexed,
    byLabel,
    completionCache: new Map<string, readonly Completion[]>(),
  }
  if (docSymbolCacheBySource.size >= MAX_DOC_SYMBOL_CACHE) {
    const oldestKey = docSymbolCacheBySource.keys().next().value
    if (oldestKey !== undefined) {
      docSymbolCacheBySource.delete(oldestKey)
    }
  }
  docSymbolCacheBySource.set(source, record)
  return record
}

export function getDocumentSymbolsForText(doc: Text): LeanDocumentSymbolCache {
  const existing = docSymbolCacheByText.get(doc)
  if (existing) return existing
  const record = getDocumentSymbolsForSource(doc.toString())
  docSymbolCacheByText.set(doc, record)
  return record
}
