import { arrayValue, objectValue, stringValue, type JsonObject } from '../../util/type-guards'

export const notebookPyrightTypeshedChunkBytes = 256 * 1024

export type NotebookPyrightTypeshedChunk = { index: number; files: Record<string, string> }

const encoder = new TextEncoder()
const chunkHeaderBytes = encodedBytes('{"files":{')
const chunkFooterBytes = encodedBytes('}}')
const pythonModuleNamePattern = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/

function encodedBytes(value: string): number {
  return encoder.encode(value).byteLength
}

function entryBytes(path: string, source: string): number {
  return encodedBytes(JSON.stringify(path)) + encodedBytes(JSON.stringify(source)) + 1
}

function emptyChunkBytes(): number {
  return chunkHeaderBytes + chunkFooterBytes
}

function ensureChunkSize(maxBytes: number, minimumBytes: number) {
  if (!Number.isInteger(maxBytes) || maxBytes < minimumBytes) {
    throw new Error(`notebook pyright chunk size must be at least ${minimumBytes} bytes`)
  }
}

export function chunkNotebookPyrightTypeshedFiles(
  files: Record<string, string>,
  maxBytes = notebookPyrightTypeshedChunkBytes,
): NotebookPyrightTypeshedChunk[] {
  ensureChunkSize(maxBytes, emptyChunkBytes() + 1)

  const chunks: NotebookPyrightTypeshedChunk[] = []
  let currentFiles: Record<string, string> = {}
  let currentBytes = emptyChunkBytes()
  let currentEntries = 0

  function flush() {
    if (currentEntries === 0) return
    chunks.push({ index: chunks.length, files: currentFiles })
    currentFiles = {}
    currentBytes = emptyChunkBytes()
    currentEntries = 0
  }

  for (const [filePath, source] of Object.entries(files).sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    const bytes = entryBytes(filePath, source)
    const nextBytes = currentBytes + bytes + (currentEntries === 0 ? 0 : 1)
    if (currentEntries > 0 && nextBytes > maxBytes) flush()
    currentFiles[filePath] = source
    currentBytes += bytes + (currentEntries === 0 ? 0 : 1)
    currentEntries += 1
  }

  flush()
  return chunks
}

export function notebookPyrightAssetManifestChunks(value: JsonObject, label: string): string[] {
  const chunks = arrayValue(value.chunks)
  if (!chunks) throw new Error(`${label} manifest is missing chunks`)
  const names: string[] = []
  for (const chunk of chunks) {
    const name = stringValue(chunk)
    if (name === undefined || name.length === 0) {
      throw new Error(`${label} manifest has an invalid chunk name`)
    }
    names.push(name)
  }
  if (names.length === 0) throw new Error(`${label} manifest has no chunks`)
  return names
}

export function notebookPyrightAssetManifestEntry(value: JsonObject, label: string): string {
  const entry = stringValue(value.entry)
  if (entry === undefined || entry.length === 0) {
    throw new Error(`${label} manifest has an invalid entry`)
  }
  return entry
}

export function notebookPyrightTypeshedFiles(value: JsonObject): Record<string, string> {
  const files = objectValue(value.files)
  if (!files) throw new Error('notebook pyright typeshed chunk is missing files')
  const result: Record<string, string> = {}
  for (const [filePath, source] of Object.entries(files)) {
    const text = stringValue(source)
    if (text === undefined) {
      throw new Error(`notebook pyright typeshed chunk has invalid source for ${filePath}`)
    }
    result[filePath] = text
  }
  return result
}

function validPythonModuleName(value: string): boolean {
  return pythonModuleNamePattern.test(value)
}

function notebookPyrightModuleStubPath(sitePackagesPath: string, moduleName: string): string {
  return `${sitePackagesPath}/${moduleName.split('.').join('/')}/__init__.pyi`
}

export function notebookPyrightPyodidePackageImports(value: JsonObject): string[] {
  const packages = objectValue(value.packages)
  if (!packages) throw new Error('pyodide package lock is missing packages')
  const names = new Set<string>()
  for (const entry of Object.values(packages)) {
    const imports = arrayValue(objectValue(entry)?.imports)
    if (!imports) continue
    for (const item of imports) {
      const name = stringValue(item)
      if (name && validPythonModuleName(name)) names.add(name)
    }
  }
  return [...names].sort((left, right) => left.localeCompare(right))
}

export function notebookPyrightPackageStubFiles(
  sitePackagesPath: string,
  moduleNames: readonly string[],
  source: string,
): Record<string, string> {
  const files: Record<string, string> = {}
  for (const moduleName of [...new Set(moduleNames)].sort((left, right) =>
    left.localeCompare(right),
  )) {
    if (!validPythonModuleName(moduleName)) continue
    files[notebookPyrightModuleStubPath(sitePackagesPath, moduleName)] = source
  }
  return files
}
