import type { CodeCell } from './types'
import { backendFor } from '../../runtime/notebook/backend'
import { isRecord } from '../type-guards'

const writefileMagicPattern = /^%%(?:writefile|file)\b(.*)$/i
const cellMagicPattern = /^%%([A-Za-z_][A-Za-z0-9_+.-]*)(?=\s|$)/
const captureCellMagics = new Set(['capture'])
const languageCellMagics = new Map<string, string>([
  ['bash', 'bash'],
  ['c', 'c'],
  ['cpp', 'cpp'],
  ['c++', 'cpp'],
  ['cxx', 'cpp'],
  ['script', 'bash'],
  ['sh', 'bash'],
  ['javascript', 'javascript'],
  ['js', 'javascript'],
  ['lean', 'lean'],
  ['lean4', 'lean'],
  ['html', 'html'],
  ['sql', 'sql'],
  ['wasm', 'wasm'],
  ['wat', 'wasm'],
])
const runtimeCellMagics = new Map<string, string>([
  ['c', 'c'],
  ['cpp', 'cpp'],
  ['c++', 'cpp'],
  ['cxx', 'cpp'],
  ['javascript', 'javascript'],
  ['js', 'javascript'],
  ['lean', 'lean'],
  ['lean4', 'lean'],
  ['wasm', 'wasm'],
  ['wat', 'wasm'],
])
const filenameLanguageExtensions = new Map<string, string>([
  ['bash', 'bash'],
  ['c', 'c'],
  ['cc', 'cpp'],
  ['cpp', 'cpp'],
  ['csv', 'csv'],
  ['cxx', 'cpp'],
  ['go', 'go'],
  ['h', 'c'],
  ['hh', 'cpp'],
  ['hpp', 'cpp'],
  ['hxx', 'cpp'],
  ['html', 'html'],
  ['java', 'java'],
  ['js', 'javascript'],
  ['json', 'json'],
  ['jsx', 'jsx'],
  ['l', 'c'],
  ['lean', 'lean'],
  ['md', 'markdown'],
  ['ml', 'ocaml'],
  ['mli', 'ocaml'],
  ['py', 'python'],
  ['rs', 'rust'],
  ['sh', 'bash'],
  ['ts', 'typescript'],
  ['tsx', 'tsx'],
  ['txt', 'text'],
  ['wat', 'wasm'],
  ['wasm', 'wasm'],
  ['zig', 'zig'],
])

function notebookCellMetadataLanguage(
  metadata: Readonly<Record<string, unknown>>,
): string | undefined {
  const vscode = isRecord(metadata.vscode) ? metadata.vscode : {}
  const language = metadata.language ?? metadata.languageId ?? vscode.languageId
  return typeof language === 'string' && language.trim() ? language.trim() : undefined
}

function filenameLanguage(filename: string): string | undefined {
  const normalized = filename.replace(/\\/g, '/').split('/').filter(Boolean).at(-1)?.toLowerCase()
  const extension = normalized?.match(/\.([a-z0-9+#-]+)$/)?.[1]
  return extension ? filenameLanguageExtensions.get(extension) : undefined
}

function writefileLanguage(line: string): string | undefined {
  const match = line.trim().match(writefileMagicPattern)
  if (!match) return undefined
  for (const part of match[1].trim().split(/\s+/)) {
    if (!part || part.startsWith('-')) continue
    return filenameLanguage(part.replace(/^["']|["']$/g, ''))
  }
}

function cellMagicLanguage(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const writefile = writefileLanguage(trimmed)
    if (writefile) return writefile
    const magic = trimmed.match(cellMagicPattern)?.[1]?.toLowerCase()
    if (magic === undefined) return undefined
    const language = languageCellMagics.get(magic)
    if (language) return language
    if (!captureCellMagics.has(magic)) return undefined
  }
}

function cellMagicRuntimeLanguage(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const magic = trimmed.match(cellMagicPattern)?.[1]?.toLowerCase()
    if (magic === undefined) return undefined
    const language = runtimeCellMagics.get(magic)
    if (language) return language
    if (!captureCellMagics.has(magic)) return undefined
  }
}

export function notebookCellDisplayLanguage(
  metadata: Readonly<Record<string, unknown>>,
  source: string,
  language: string,
): string {
  return cellMagicLanguage(source) ?? notebookCellMetadataLanguage(metadata) ?? language
}

export function notebookCellRuntimeBackend(cell: CodeCell, notebookLanguage: string) {
  const runtimeLanguage = cellMagicRuntimeLanguage(cell.source)
  if (runtimeLanguage !== undefined) {
    const backend = backendFor(runtimeLanguage)
    if (backend) return backend
  }
  return backendFor(notebookLanguage)
}
