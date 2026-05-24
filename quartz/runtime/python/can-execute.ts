const browserRuntimeExtensionDirectives = new Set(['autoreload', 'nb_mypy'])
const browserRuntimeThreadingReason =
  'Python threading and multiprocessing are unavailable in the browser runtime because Pyodide does not support starting threads or processes. Use QUARTZ_NOTEBOOK_MODE=execute or a server Python runtime for this cell.'
const wat2wasmFeatureOptions = new Set([
  '--enable-annotations',
  '--enable-bulk-memory',
  '--enable-code-metadata',
  '--enable-exceptions',
  '--enable-extended-const',
  '--enable-function-references',
  '--enable-gc',
  '--enable-memory64',
  '--enable-multi-memory',
  '--enable-multi-value',
  '--enable-mutable-globals',
  '--enable-reference-types',
  '--enable-relaxed-simd',
  '--enable-saturating-float-to-int',
  '--enable-sign-extension',
  '--enable-simd',
  '--enable-tail-call',
  '--enable-threads',
])
const notebookLsOptionChars = new Set(['1', 'a', 'h', 'l'])

function notebookShellWords(value: string): string[] {
  const words: string[] = []
  let word = ''
  let quote = ''
  let escaped = false
  for (const char of value) {
    if (escaped) {
      word += char
      escaped = false
      continue
    }
    if (char === '\\') {
      escaped = true
      continue
    }
    if (quote.length > 0) {
      if (char === quote) quote = ''
      else word += char
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (word.length > 0) {
        words.push(word)
        word = ''
      }
      continue
    }
    word += char
  }
  if (escaped) word += '\\'
  if (word.length > 0) words.push(word)
  return words
}

function notebookSandboxPathReason(
  command: string,
  path: string,
  allowDot = false,
): string | undefined {
  const normalized = path.trim().replace(/\\/g, '/')
  if (normalized.length === 0 || normalized.includes('\0')) {
    return `${command} path is unavailable in the browser runtime sandbox`
  }
  if (normalized.startsWith('/')) {
    return `${command} path ${path} is outside the browser runtime sandbox`
  }
  const parts = normalized.split('/')
  if (parts.some(part => part === '..')) {
    return `${command} path ${path} is outside the browser runtime sandbox`
  }
  if (!allowDot && (normalized === '.' || parts.every(part => part === '' || part === '.'))) {
    return `${command} path ${path} is unavailable in the browser runtime sandbox`
  }
}

function notebookExtensionDirective(line: string): string | undefined {
  const match = line.match(/^%?(?:load_ext|reload_ext)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$/)
  return match?.[1]?.toLowerCase()
}

function browserRuntimeDirectiveReason(line: string): string | undefined {
  const extension = notebookExtensionDirective(line)
  if (extension !== undefined) {
    return browserRuntimeExtensionDirectives.has(extension)
      ? undefined
      : `IPython extension ${extension} is unavailable in the browser runtime`
  }
  if (/^%autoreload(?:\s|$)/.test(line)) return undefined
  if (/^%matplotlib(?:\s|$)/.test(line)) return undefined
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function pythonImportAliases(source: string, moduleName: string): Set<string> {
  const aliases = new Set<string>()
  const escapedModuleName = escapeRegExp(moduleName)
  for (const line of source.split(/\r?\n/)) {
    const withoutComment = line.replace(/#.*/, '')
    for (const importMatch of withoutComment.matchAll(/(?:^|[;:])\s*import\s+([^;]+)/g)) {
      for (const part of importMatch[1].split(',')) {
        const match = part
          .trim()
          .match(new RegExp(`^${escapedModuleName}\\s*(?:as\\s+([A-Za-z_][A-Za-z0-9_]*))?$`))
        if (match) aliases.add(match[1] ?? moduleName.split('.').at(-1) ?? moduleName)
      }
    }
  }
  return aliases
}

function pythonFromImportNames(source: string, moduleName: string): Set<string> {
  const names = new Set<string>()
  const escapedModuleName = escapeRegExp(moduleName)
  for (const line of source.split(/\r?\n/)) {
    const withoutComment = line.replace(/#.*/, '')
    for (const match of withoutComment.matchAll(
      new RegExp(`(?:^|[;:])\\s*from\\s+${escapedModuleName}\\s+import\\s+([^;]+)`, 'g'),
    )) {
      for (const part of match[1].split(',')) {
        const name = part
          .trim()
          .split(/\s+as\s+/)[0]
          ?.trim()
        if (name) names.add(name)
      }
    }
  }
  return names
}

function hasQualifiedPythonUse(source: string, aliases: Set<string>, members: string[]): boolean {
  for (const alias of aliases) {
    const escaped = escapeRegExp(alias)
    if (new RegExp(`\\b${escaped}\\s*\\.\\s*(?:${members.join('|')})\\b`).test(source)) return true
  }
  return false
}

function browserRuntimeThreadingReasonForSource(source: string): string | undefined {
  const threadingNames = pythonFromImportNames(source, 'threading')
  if (threadingNames.has('*') || threadingNames.has('Thread')) return browserRuntimeThreadingReason
  const multiprocessingNames = pythonFromImportNames(source, 'multiprocessing')
  if (multiprocessingNames.has('*') || multiprocessingNames.has('Process')) {
    return browserRuntimeThreadingReason
  }
  const futuresNames = pythonFromImportNames(source, 'concurrent.futures')
  if (
    futuresNames.has('*') ||
    futuresNames.has('ThreadPoolExecutor') ||
    futuresNames.has('ProcessPoolExecutor')
  ) {
    return browserRuntimeThreadingReason
  }
  if (hasQualifiedPythonUse(source, pythonImportAliases(source, 'threading'), ['Thread'])) {
    return browserRuntimeThreadingReason
  }
  if (hasQualifiedPythonUse(source, pythonImportAliases(source, 'multiprocessing'), ['Process'])) {
    return browserRuntimeThreadingReason
  }
  if (
    hasQualifiedPythonUse(source, pythonImportAliases(source, 'concurrent.futures'), [
      'ThreadPoolExecutor',
      'ProcessPoolExecutor',
    ])
  ) {
    return browserRuntimeThreadingReason
  }
}

function notebookWriteFileDirectiveReason(source: string): { handled: boolean; reason?: string } {
  const first = source.split(/\r?\n/, 1)[0]?.trim() ?? ''
  if (!first.startsWith('%%writefile')) return { handled: false }
  const words = first
    .replace(/^%%writefile\b/, '')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
  let filename = ''
  for (const word of words) {
    if (word === '-a') continue
    if (word.startsWith('-')) {
      return {
        handled: true,
        reason: `%%writefile option ${word} is unavailable in the browser runtime`,
      }
    }
    if (filename) return { handled: true, reason: '%%writefile accepts one file name' }
    filename = word
  }
  if (!filename) return { handled: true, reason: '%%writefile requires a file name' }
  return { handled: true, reason: notebookSandboxPathReason('%%writefile', filename) }
}

function notebookShellDirectiveReason(line: string): { handled: boolean; reason?: string } {
  if (!line.startsWith('!')) return { handled: false }
  const command = line.slice(1).trim()
  if (/^cat(?:\s|$)/.test(command)) {
    const words = notebookShellWords(command)
    if (words.length === 1) return { handled: true, reason: 'cat requires a file' }
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]
      if (word.startsWith('-')) {
        return { handled: true, reason: 'cat options are unavailable in the browser runtime' }
      }
      const pathReason = notebookSandboxPathReason('cat', word)
      if (pathReason) return { handled: true, reason: pathReason }
    }
    return { handled: true }
  }
  if (/^ls(?:\s|$)/.test(command)) {
    const words = notebookShellWords(command)
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]
      if (word.startsWith('-')) {
        if (
          word === '-' ||
          Array.from(word.slice(1)).some(char => !notebookLsOptionChars.has(char))
        ) {
          return {
            handled: true,
            reason: `ls option ${word} is unavailable in the browser runtime`,
          }
        }
        continue
      }
      const pathReason = notebookSandboxPathReason('ls', word, true)
      if (pathReason) return { handled: true, reason: pathReason }
    }
    return { handled: true }
  }
  if (/^(?:wat2wasm|wasm2wat)(?:\s|$)/.test(command)) {
    const name = command.split(/\s+/, 1)[0] ?? 'wabt'
    const words = notebookShellWords(command)
    let input = ''
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]
      if (word === '-o' || word === '--output') {
        index += 1
        if (!words[index]) return { handled: true, reason: `${word} requires a file name` }
        const pathReason = notebookSandboxPathReason(name, words[index])
        if (pathReason) return { handled: true, reason: pathReason }
        continue
      }
      if (word.startsWith('-o') && word.length > 2) {
        const pathReason = notebookSandboxPathReason(name, word.slice(2))
        if (pathReason) return { handled: true, reason: pathReason }
        continue
      }
      if (word.startsWith('--output=')) {
        const pathReason = notebookSandboxPathReason(name, word.slice('--output='.length))
        if (pathReason) return { handled: true, reason: pathReason }
        continue
      }
      if (word.startsWith('--enable-')) {
        if (wat2wasmFeatureOptions.has(word)) continue
        return {
          handled: true,
          reason: `${name} option ${word} is unavailable in the browser runtime`,
        }
      }
      if (word.startsWith('--disable-')) {
        const enabled = `--enable-${word.slice('--disable-'.length)}`
        if (wat2wasmFeatureOptions.has(enabled)) continue
        return {
          handled: true,
          reason: `${name} option ${word} is unavailable in the browser runtime`,
        }
      }
      if (name === 'wasm2wat' && (word === '--fold-exprs' || word === '--inline-exports')) continue
      if (word.startsWith('-')) {
        return {
          handled: true,
          reason: `${name} option ${word} is unavailable in the browser runtime`,
        }
      }
      if (input) return { handled: true, reason: `${name} accepts one input file` }
      const pathReason = notebookSandboxPathReason(name, word)
      if (pathReason) return { handled: true, reason: pathReason }
      input = word
    }
    return input ? { handled: true } : { handled: true, reason: `${name} requires an input file` }
  }
  return { handled: true, reason: 'shell escapes are unavailable in the browser runtime' }
}

export function unsupportedNotebookRuntimeReason(source: string): string | undefined {
  const threading = browserRuntimeThreadingReasonForSource(source)
  if (threading) return threading
  const writeFile = notebookWriteFileDirectiveReason(source)
  if (writeFile.handled) return writeFile.reason
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (trimmed.length === 0) continue
    if (/^%pip\s+install(?:\s|$)/.test(trimmed)) continue
    if (/^%timeit(?:\s|$)/.test(trimmed)) continue
    if (/^%time(?:\s|$)/.test(trimmed)) continue
    if (/^!(?:pip|uv\s+pip|python3?\s+-m\s+pip)\s+install(?:\s|$)/.test(trimmed)) continue
    const directiveReason = browserRuntimeDirectiveReason(trimmed)
    if (directiveReason !== undefined) return directiveReason
    if (notebookExtensionDirective(trimmed) !== undefined) continue
    if (/^%autoreload(?:\s|$)/.test(trimmed)) continue
    if (/^%matplotlib(?:\s|$)/.test(trimmed)) continue
    const shell = notebookShellDirectiveReason(trimmed)
    if (shell.handled) return shell.reason
    if (trimmed.startsWith('%%')) return 'cell magics are unavailable in the browser runtime'
    if (trimmed.startsWith('%')) return 'IPython magics are unavailable in the browser runtime'
    if (trimmed.startsWith('!')) return 'shell escapes are unavailable in the browser runtime'
  }
}
