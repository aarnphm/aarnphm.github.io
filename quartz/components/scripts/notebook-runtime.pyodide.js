import loadWabt from 'wabt'
import { ensureNotebookMlRuntime, installNotebookMlBridge } from './notebook-runtime.ml.js'
import notebookRuntimeBootstrap from './notebook-runtime.pyodide.py'

const source = 'quartz-notebook-runtime'
let runtimeId = ''
let pyodide
let currentCellId = ''
let assetSequence = 0
const pendingAssets = new Map()
const streamBuffers = new Map()
const stdoutDecoder = new TextDecoder()
const stderrDecoder = new TextDecoder()
let stdlibModules
let debugEnabled = false
let lastPythonError
let wabtRuntime
const notebookFilesystemRoot = '/quartz-notebook'
const loadedPackageRequests = new Set()
const ignoredImportPackages = new Set([
  'import_ipynb',
  'ipython',
  'jax',
  'js',
  'nbimporter',
  'pyodide',
  'torch',
])
const pipOptionsWithValues = new Set([
  '--abi',
  '--constraint',
  '--extra-index-url',
  '--find-links',
  '--implementation',
  '--index-url',
  '--no-binary',
  '--only-binary',
  '--platform',
  '--prefix',
  '--python',
  '--root',
  '--src',
  '--target',
  '--trusted-host',
  '-c',
])
const pyodideImportPackages = new Map([
  ['altair', 'altair'],
  ['astropy', 'astropy'],
  ['beautifulsoup4', 'beautifulsoup4'],
  ['bs4', 'beautifulsoup4'],
  ['cv2', 'opencv-python'],
  ['fiona', 'fiona'],
  ['geopandas', 'geopandas'],
  ['h5py', 'h5py'],
  ['html5lib', 'html5lib'],
  ['httpx', 'httpx'],
  ['jinja2', 'Jinja2'],
  ['lxml', 'lxml'],
  ['matplotlib', 'matplotlib'],
  ['mpl_toolkits', 'matplotlib'],
  ['networkx', 'networkx'],
  ['nltk', 'nltk'],
  ['numpy', 'numpy'],
  ['openai', 'openai'],
  ['opencv-python', 'opencv-python'],
  ['pandas', 'pandas'],
  ['pil', 'Pillow'],
  ['plotly', 'plotly'],
  ['pyarrow', 'pyarrow'],
  ['pydantic', 'pydantic'],
  ['pygame', 'pygame-ce'],
  ['requests', 'requests'],
  ['scipy', 'scipy'],
  ['scikit-image', 'scikit-image'],
  ['scikit-learn', 'scikit-learn'],
  ['seaborn', 'seaborn'],
  ['skimage', 'scikit-image'],
  ['sklearn', 'scikit-learn'],
  ['statsmodels', 'statsmodels'],
  ['sympy', 'sympy'],
  ['pyyaml', 'pyyaml'],
  ['yaml', 'pyyaml'],
])
const runtimeProvidedPackages = new Set(['jax', 'torch'])
const browserRuntimeExtensionDirectives = new Set(['autoreload', 'nb_mypy'])
const unsupportedNativePackages = new Map([
  [
    'flax',
    'Flax depends on JAX and jaxlib, which require a native XLA runtime outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'jaxlib',
    'jaxlib requires native XLA runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'keras',
    'Keras depends on native TensorFlow/JAX/PyTorch runtimes unavailable in this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'optax',
    'Optax depends on JAX and jaxlib, which require a native XLA runtime outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'tensorflow',
    'TensorFlow requires native runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'torchaudio',
    'torchaudio depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'torchtext',
    'torchtext depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'torchvision',
    'torchvision depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
  [
    'triton',
    'Triton requires native compiler/runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  ],
])
const wat2wasmFeatureFlags = new Map([
  ['--enable-annotations', 'annotations'],
  ['--enable-bulk-memory', 'bulk_memory'],
  ['--enable-code-metadata', 'code_metadata'],
  ['--enable-exceptions', 'exceptions'],
  ['--enable-extended-const', 'extended_const'],
  ['--enable-function-references', 'function_references'],
  ['--enable-gc', 'gc'],
  ['--enable-memory64', 'memory64'],
  ['--enable-multi-memory', 'multi_memory'],
  ['--enable-multi-value', 'multi_value'],
  ['--enable-mutable-globals', 'mutable_globals'],
  ['--enable-reference-types', 'reference_types'],
  ['--enable-relaxed-simd', 'relaxed_simd'],
  ['--enable-saturating-float-to-int', 'sat_float_to_int'],
  ['--enable-sign-extension', 'sign_extension'],
  ['--enable-simd', 'simd'],
  ['--enable-tail-call', 'tail_call'],
  ['--enable-threads', 'threads'],
])
const notebookLsOptionChars = new Set(['1', 'a', 'h', 'l'])
function post(message, transfer) {
  globalThis.postMessage({ source, runtimeId, ...message }, transfer || [])
}
function setRuntimeStatus(text) {
  post({ type: 'status', text })
}
function textOf(value) {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  try {
    return String(value)
  } catch {
    return Object.prototype.toString.call(value)
  }
}
function notebookSandboxPathError(command, path, allowDot) {
  const normalized = textOf(path).trim().replaceAll('\\', '/')
  if (!normalized || normalized.includes('\0')) {
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
function notebookSandboxPath(command, path, allowDot) {
  const reason = notebookSandboxPathError(command, path, allowDot)
  if (reason) throw new Error(reason)
  const parts = textOf(path)
    .trim()
    .replaceAll('\\', '/')
    .split('/')
    .filter(part => part && part !== '.')
  return `${notebookFilesystemRoot}/${parts.join('/')}`
}
function emitOutput(output) {
  if (!currentCellId) return
  emitOutputForCell(currentCellId, output)
}
function emitOutputForCell(cellId, output) {
  if (!cellId) return
  post({ type: 'output', cellId, output })
}
function bufferStreamForCell(cellId, name, text) {
  if (!cellId || !text) return
  const key = cellId + '\u0000' + name
  streamBuffers.set(key, (streamBuffers.get(key) || '') + text)
}
function bufferStreamBytesForCell(cellId, name, bytes, decoder) {
  if (!bytes) return 0
  const text = decoder.decode(bytes, { stream: true })
  bufferStreamForCell(cellId, name, text)
  return bytes.length
}
function flushStreamDecoderForCell(cellId, name, decoder) {
  bufferStreamForCell(cellId, name, decoder.decode())
}
function flushStreamsForCell(cellId) {
  flushStreamDecoderForCell(cellId, 'stdout', stdoutDecoder)
  flushStreamDecoderForCell(cellId, 'stderr', stderrDecoder)
  for (const [key, text] of Array.from(streamBuffers.entries())) {
    const [owner, name] = key.split('\u0000')
    if (owner !== cellId) continue
    streamBuffers.delete(key)
    if (text) emitOutputForCell(cellId, { type: 'stream', name, text })
  }
}
function debugOutput(phase, cellId, error) {
  return {
    phase,
    cellId,
    errorName: error && error.name ? textOf(error.name) : 'Error',
    errorMessage: textOf(error),
    stack: error && error.stack ? textOf(error.stack) : undefined,
  }
}
function emitError(error, phase) {
  const text = textOf(error)
  const output = {
    type: 'error',
    ename: error && error.name ? textOf(error.name) : 'Error',
    evalue: text,
    traceback: error && error.stack ? textOf(error.stack) : text,
  }
  if (debugEnabled) output.debug = debugOutput(phase || 'runtime', currentCellId, error)
  emitOutput(output)
}
function emitPythonError(error, phase) {
  if (!lastPythonError) {
    emitError(error, phase)
    return
  }
  const output = {
    type: 'error',
    ename: lastPythonError.ename,
    evalue: lastPythonError.evalue,
    traceback: lastPythonError.traceback,
  }
  if (debugEnabled) output.debug = debugOutput(phase || 'python', currentCellId, error)
  lastPythonError = undefined
  emitOutput(output)
}
function handlePythonError(serialized) {
  try {
    const value = JSON.parse(textOf(serialized))
    if (
      value &&
      typeof value.ename === 'string' &&
      typeof value.evalue === 'string' &&
      typeof value.traceback === 'string'
    ) {
      lastPythonError = value
    }
  } catch {
    lastPythonError = undefined
  }
}
function sandboxFetch(input, cellId) {
  const url = typeof input === 'string' ? input : input && input.url
  if (typeof url !== 'string') return Promise.reject(new Error('unsupported fetch input'))
  const assetId = 'asset-' + ++assetSequence
  post({ type: 'asset', cellId, assetId, url })
  return new Promise((resolve, reject) => {
    pendingAssets.set(assetId, { resolve, reject, cellId })
  }).catch(error => {
    const output = {
      type: 'error',
      ename: error && error.name ? textOf(error.name) : 'AssetError',
      evalue: textOf(error),
      traceback: error && error.stack ? textOf(error.stack) : textOf(error),
    }
    if (debugEnabled) output.debug = debugOutput('asset', cellId, error)
    emitOutputForCell(cellId, output)
    throw error
  })
}
async function executeDisplayJavascript(code) {
  const cellId = currentCellId
  if (!cellId) return
  post({ type: 'display-javascript', cellId, code: textOf(code) })
}
function handleDisplayPayload(serialized) {
  let data
  try {
    data = JSON.parse(serialized)
  } catch (error) {
    emitError(error)
    return
  }
  if (data['text/html']) {
    emitOutput({ type: 'html', html: textOf(data['text/html']) })
  } else if (data['application/javascript']) {
    executeDisplayJavascript(textOf(data['application/javascript'])).catch(emitError)
  } else if (data['text/plain']) {
    emitOutput({ type: 'text', text: textOf(data['text/plain']) })
  }
}
function shellWords(value) {
  const words = []
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
    if (quote) {
      if (char === quote) {
        quote = ''
      } else {
        word += char
      }
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (word) {
        words.push(word)
        word = ''
      }
      continue
    }
    word += char
  }
  if (escaped) word += '\\'
  if (word) words.push(word)
  return words
}
function pipInstallDirective(line) {
  const trimmed = line.trim()
  const prefixes = [/^%pip\s+/, /^!pip\s+/, /^!uv\s+pip\s+/, /^!python3?\s+-m\s+pip\s+/]
  let command
  for (const prefix of prefixes) {
    if (!prefix.test(trimmed)) continue
    command = trimmed.replace(prefix, '')
    break
  }
  if (command === undefined) return undefined
  const words = shellWords(command)
  if (words[0] !== 'install') {
    return { requirements: [], error: 'only pip install is available in the browser runtime' }
  }
  const requirements = []
  for (let i = 1; i < words.length; i++) {
    const word = words[i]
    if (!word) continue
    if (
      word === '-r' ||
      word === '--requirement' ||
      word.startsWith('-r') ||
      word.startsWith('--requirement=')
    ) {
      return {
        requirements: [],
        error: 'requirements files are unavailable in the browser runtime',
      }
    }
    if (pipOptionsWithValues.has(word)) {
      i += 1
      continue
    }
    if (word.startsWith('--') && word.includes('=')) continue
    if (word.startsWith('-')) continue
    requirements.push(word)
  }
  return { requirements }
}
function notebookExtensionDirective(line) {
  const match = line.trim().match(/^%?(?:load_ext|reload_ext)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$/)
  return match ? match[1].toLowerCase() : undefined
}
function browserRuntimeDirective(line) {
  const trimmed = line.trim()
  const extension = notebookExtensionDirective(trimmed)
  if (extension !== undefined) {
    if (!browserRuntimeExtensionDirectives.has(extension)) {
      return { error: `IPython extension ${extension} is unavailable in the browser runtime` }
    }
    return {
      warning:
        extension === 'nb_mypy'
          ? 'nb_mypy is unavailable in the browser runtime; continuing without notebook type checking.'
          : undefined,
    }
  }
  if (/^%autoreload(?:\s|$)/.test(trimmed)) return {}
  if (/^%matplotlib(?:\s|$)/.test(trimmed)) return {}
}
function writeFileDirective(code) {
  const lines = code.split(/\r?\n/)
  const first = lines[0]?.trim() ?? ''
  if (!first.startsWith('%%writefile')) return undefined
  const words = shellWords(first.replace(/^%%writefile\b/, '').trim())
  let append = false
  let filename = ''
  for (const word of words) {
    if (word === '-a') {
      append = true
      continue
    }
    if (word.startsWith('-')) {
      return { error: `%%writefile option ${word} is unavailable in the browser runtime` }
    }
    if (filename) return { error: '%%writefile accepts one file name' }
    filename = word
  }
  if (!filename) return { error: '%%writefile requires a file name' }
  const pathError = notebookSandboxPathError('%%writefile', filename)
  if (pathError) return { error: pathError }
  return { filename, append, content: lines.slice(1).join('\n') }
}
function defaultWasmOutputPath(input) {
  return input.replace(/(?:\.wat)?$/i, '.wasm')
}
function wabtFeatureOption(name, word) {
  const feature = wat2wasmFeatureFlags.get(word)
  if (feature) return { feature, enabled: true }
  if (!word.startsWith('--disable-')) return undefined
  const enabledFlag = `--enable-${word.slice('--disable-'.length)}`
  const disabledFeature = wat2wasmFeatureFlags.get(enabledFlag)
  if (!disabledFeature)
    return { error: `${name} option ${word} is unavailable in the browser runtime` }
  return { feature: disabledFeature, enabled: false }
}
function wat2wasmDirective(command) {
  const words = shellWords(command)
  if (words[0] !== 'wat2wasm') return undefined
  let input = ''
  let output = ''
  const features = {}
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index]
    if (word === '-o' || word === '--output') {
      index += 1
      if (!words[index]) return { error: `${word} requires a file name` }
      const pathError = notebookSandboxPathError('wat2wasm', words[index])
      if (pathError) return { error: pathError }
      output = words[index]
      continue
    }
    if (word.startsWith('-o') && word.length > 2) {
      const pathError = notebookSandboxPathError('wat2wasm', word.slice(2))
      if (pathError) return { error: pathError }
      output = word.slice(2)
      continue
    }
    if (word.startsWith('--output=')) {
      const pathError = notebookSandboxPathError('wat2wasm', word.slice('--output='.length))
      if (pathError) return { error: pathError }
      output = word.slice('--output='.length)
      continue
    }
    const feature = wabtFeatureOption('wat2wasm', word)
    if (feature) {
      if (feature.error) return { error: feature.error }
      features[feature.feature] = feature.enabled
      continue
    }
    if (word.startsWith('-')) {
      return { error: `wat2wasm option ${word} is unavailable in the browser runtime` }
    }
    if (input) return { error: 'wat2wasm accepts one input file' }
    const pathError = notebookSandboxPathError('wat2wasm', word)
    if (pathError) return { error: pathError }
    input = word
  }
  if (!input) return { error: 'wat2wasm requires an input file' }
  return { input, output: output || defaultWasmOutputPath(input), features }
}
function wasm2watDirective(command) {
  const words = shellWords(command)
  if (words[0] !== 'wasm2wat') return undefined
  let input = ''
  let output = ''
  const features = {}
  const textOptions = { foldExprs: false, inlineExport: false }
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index]
    if (word === '-o' || word === '--output') {
      index += 1
      if (!words[index]) return { error: `${word} requires a file name` }
      const pathError = notebookSandboxPathError('wasm2wat', words[index])
      if (pathError) return { error: pathError }
      output = words[index]
      continue
    }
    if (word.startsWith('-o') && word.length > 2) {
      const pathError = notebookSandboxPathError('wasm2wat', word.slice(2))
      if (pathError) return { error: pathError }
      output = word.slice(2)
      continue
    }
    if (word.startsWith('--output=')) {
      const pathError = notebookSandboxPathError('wasm2wat', word.slice('--output='.length))
      if (pathError) return { error: pathError }
      output = word.slice('--output='.length)
      continue
    }
    if (word === '--fold-exprs') {
      textOptions.foldExprs = true
      continue
    }
    if (word === '--inline-exports') {
      textOptions.inlineExport = true
      continue
    }
    const feature = wabtFeatureOption('wasm2wat', word)
    if (feature) {
      if (feature.error) return { error: feature.error }
      features[feature.feature] = feature.enabled
      continue
    }
    if (word.startsWith('-')) {
      return { error: `wasm2wat option ${word} is unavailable in the browser runtime` }
    }
    if (input) return { error: 'wasm2wat accepts one input file' }
    const pathError = notebookSandboxPathError('wasm2wat', word)
    if (pathError) return { error: pathError }
    input = word
  }
  if (!input) return { error: 'wasm2wat requires an input file' }
  return { input, output, features, textOptions }
}
function lsDirective(command) {
  const words = shellWords(command)
  if (words[0] !== 'ls') return undefined
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index]
    if (!word.startsWith('-')) continue
    if (word === '-' || Array.from(word.slice(1)).some(char => !notebookLsOptionChars.has(char))) {
      return { error: `ls option ${word} is unavailable in the browser runtime` }
    }
    continue
  }
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index]
    if (word.startsWith('-')) continue
    const pathError = notebookSandboxPathError('ls', word, true)
    if (pathError) return { error: pathError }
  }
  return { command }
}
function catDirective(command) {
  const words = shellWords(command)
  if (words[0] !== 'cat') return undefined
  if (words.length === 1) return { error: 'cat requires a file' }
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index]
    if (word.startsWith('-')) return { error: 'cat options are unavailable in the browser runtime' }
    const pathError = notebookSandboxPathError('cat', word)
    if (pathError) return { error: pathError }
  }
  return { command }
}
function shellDirective(line) {
  const trimmed = line.trim()
  if (!trimmed.startsWith('!')) return undefined
  const command = trimmed.slice(1).trim()
  const cat = catDirective(command)
  if (cat) return cat.error ? { error: cat.error } : { command: cat.command }
  const ls = lsDirective(command)
  if (ls) return ls.error ? { error: ls.error } : { command: ls.command }
  const wat2wasm = wat2wasmDirective(command)
  if (wat2wasm) return wat2wasm.error ? { error: wat2wasm.error } : { wat2wasm }
  const wasm2wat = wasm2watDirective(command)
  if (wasm2wat) return wasm2wat.error ? { error: wasm2wat.error } : { wasm2wat }
  return { error: 'shell escapes are unavailable in the browser runtime' }
}
function stripPackageDirectives(code) {
  const writeFile = writeFileDirective(code)
  if (writeFile) {
    if (writeFile.error) throw new Error(writeFile.error)
    return {
      code: `__quartz_writefile(${JSON.stringify(writeFile.filename)}, ${JSON.stringify(
        writeFile.content,
      )}, ${writeFile.append ? 'True' : 'False'})`,
      requirements: [],
      shellCommands: [],
      warnings: [],
    }
  }
  const lines = []
  const requirements = []
  const shellCommands = []
  const warnings = []
  for (const line of code.split(/\r?\n/)) {
    const directive = pipInstallDirective(line)
    if (directive) {
      if (directive.error) throw new Error(directive.error)
      requirements.push(...directive.requirements)
      continue
    }
    const runtimeDirective = browserRuntimeDirective(line)
    if (runtimeDirective) {
      if (runtimeDirective.error) throw new Error(runtimeDirective.error)
      if (runtimeDirective.warning) warnings.push(runtimeDirective.warning)
      continue
    }
    const shell = shellDirective(line)
    if (shell) {
      if (shell.error) throw new Error(shell.error)
      if (shell.wat2wasm) {
        shellCommands.push({ type: 'wat2wasm', ...shell.wat2wasm })
        continue
      }
      if (shell.wasm2wat) {
        shellCommands.push({ type: 'wasm2wat', ...shell.wasm2wat })
        continue
      }
      lines.push(`__quartz_shell(${JSON.stringify(shell.command)})`)
      continue
    }
    lines.push(line)
  }
  return { code: lines.join('\n'), requirements, shellCommands, warnings }
}
function packageRootName(requirement) {
  const match = requirement.trim().match(/^['"]?([A-Za-z0-9_.-]+)/)
  return match ? match[1].toLowerCase().replace(/_/g, '-') : ''
}
function unsupportedNativeMessageForPackage(name) {
  const normalized = name.toLowerCase().replace(/_/g, '-')
  return (
    unsupportedNativePackages.get(normalized) ||
    unsupportedNativePackages.get(normalized.split('-')[0])
  )
}
function isRuntimeProvidedPackage(name) {
  const normalized = name.toLowerCase().replace(/_/g, '-')
  return (
    runtimeProvidedPackages.has(normalized) || runtimeProvidedPackages.has(normalized.split('-')[0])
  )
}
function importNames(code) {
  const names = new Set()
  for (const line of code.split(/\r?\n/)) {
    const withoutComment = line.replace(/#.*/, '')
    for (const importMatch of withoutComment.matchAll(/(?:^|[;:])\s*import\s+([^;]+)/g)) {
      for (const part of importMatch[1].split(',')) {
        const name = part
          .trim()
          .split(/\s+|\./)[0]
          ?.replace(/\W+$/, '')
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) names.add(name)
      }
    }
    for (const fromMatch of withoutComment.matchAll(
      /(?:^|[;:])\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\b/g,
    )) {
      names.add(fromMatch[1])
    }
  }
  return [...names]
}
function stdlibSet(runtime) {
  if (stdlibModules) return stdlibModules
  try {
    const modules = runtime.runPython('import sys; list(sys.stdlib_module_names)')
    try {
      stdlibModules = new Set(modules.toJs())
    } finally {
      if (modules && typeof modules.destroy === 'function') modules.destroy()
    }
  } catch {
    stdlibModules = new Set([
      'abc',
      'argparse',
      'ast',
      'collections',
      'contextlib',
      'dataclasses',
      'functools',
      'importlib',
      'itertools',
      'json',
      'math',
      'os',
      'pathlib',
      'random',
      're',
      'statistics',
      'string',
      'sys',
      'textwrap',
      'time',
      'types',
      'typing',
    ])
  }
  return stdlibModules
}
function packageNameForImport(runtime, name) {
  const root = name.split('.')[0]
  const normalized = root.toLowerCase()
  if (ignoredImportPackages.has(normalized)) return undefined
  const stdlib = stdlibSet(runtime)
  if (stdlib.has(root) || stdlib.has(normalized)) return undefined
  if (unsupportedNativePackages.has(normalized)) return undefined
  return pyodideImportPackages.get(normalized)
}
async function loadPyodideImportPackages(runtime, code) {
  const packages = []
  const seen = new Set()
  for (const name of importNames(code)) {
    const packageName = packageNameForImport(runtime, name)
    if (
      !packageName ||
      seen.has(packageName) ||
      loadedPackageRequests.has('pyodide:' + packageName)
    ) {
      continue
    }
    seen.add(packageName)
    packages.push(packageName)
  }
  if (packages.length === 0) return
  setRuntimeStatus('loading ' + packages.join(', '))
  await runtime.loadPackage(packages)
  for (const packageName of packages) loadedPackageRequests.add('pyodide:' + packageName)
}
async function installPipRequirements(runtime, requirements) {
  const packages = []
  const seen = new Set()
  for (const requirement of requirements) {
    const packageName = requirement.trim()
    if (!packageName) continue
    if (isRuntimeProvidedPackage(packageRootName(packageName))) {
      loadedPackageRequests.add('runtime:' + packageRootName(packageName))
      continue
    }
    const nativeMessage = unsupportedNativeMessageForPackage(packageRootName(packageName))
    if (nativeMessage) throw new Error(nativeMessage)
    const key = 'pip:' + packageName
    if (seen.has(key) || loadedPackageRequests.has(key)) continue
    seen.add(key)
    packages.push(packageName)
  }
  if (packages.length === 0) return
  setRuntimeStatus('installing ' + packages.join(', '))
  if (!loadedPackageRequests.has('pyodide:micropip')) {
    await runtime.loadPackage('micropip')
    loadedPackageRequests.add('pyodide:micropip')
  }
  const micropip = runtime.pyimport('micropip')
  try {
    await micropip.install(packages)
  } finally {
    if (micropip && typeof micropip.destroy === 'function') micropip.destroy()
  }
  for (const packageName of packages) {
    loadedPackageRequests.add('pip:' + packageName)
    const pyodidePackage = pyodideImportPackages.get(packageRootName(packageName))
    if (pyodidePackage) loadedPackageRequests.add('pyodide:' + pyodidePackage)
  }
}
async function preparePackages(runtime, code, requirements, modules) {
  await installPipRequirements(runtime, requirements)
  const moduleCode = Array.isArray(modules)
    ? modules
        .map(module => (module && typeof module.source === 'string' ? module.source : ''))
        .join('\n')
    : ''
  await loadPyodideImportPackages(runtime, code + '\n' + moduleCode)
}
function usesNotebookMlRuntime(code, modules) {
  const names = new Set(importNames(code))
  if (Array.isArray(modules)) {
    for (const module of modules) {
      if (!module || typeof module.source !== 'string') continue
      for (const name of importNames(module.source)) names.add(name)
    }
  }
  return (
    names.has('jax') ||
    names.has('torch') ||
    /\b(?:jax|jnp|torch|value_and_grad|tree_util)\b/.test(code)
  )
}
function timeitDirective(line) {
  const match = line.match(/^(\s*)%timeit\b(.*)$/)
  if (!match) return undefined
  let rest = match[2].trim()
  let number = null
  let repeat = null
  rest = rest.replace(/(?:^|\s)-n\s*(\d+)/, (_all, value) => {
    number = Number(value)
    return ' '
  })
  rest = rest.replace(/(?:^|\s)-r\s*(\d+)/, (_all, value) => {
    repeat = Number(value)
    return ' '
  })
  const statement = rest.trim()
  if (!statement) throw new Error('%timeit requires a statement')
  return `${match[1]}__quartz_timeit(${JSON.stringify(statement)}, globals(), locals(), ${
    number ?? 'None'
  }, ${repeat ?? 'None'})`
}
function translateLineMagics(code) {
  return code
    .split(/\r?\n/)
    .map(line => timeitDirective(line) ?? line)
    .join('\n')
}
function notebookPath(filename) {
  return notebookSandboxPath('notebook', filename, false)
}
function contentTypeForPath(path) {
  if (/\.wasm$/i.test(path)) return 'application/wasm'
  if (/\.wat$/i.test(path)) return 'text/plain'
  return 'application/octet-stream'
}
function copyArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}
function readRuntimeBinaryFile(path) {
  if (!pyodide) throw new Error('runtime filesystem is unavailable')
  const normalized = notebookPath(path)
  const bytes = pyodide.FS.readFile(normalized)
  return {
    path: normalized,
    contentType: contentTypeForPath(normalized),
    bytes: copyArrayBuffer(bytes),
  }
}
function readRuntimeTextFile(path) {
  if (!pyodide) throw new Error('runtime filesystem is unavailable')
  return pyodide.FS.readFile(notebookPath(path), { encoding: 'utf8' })
}
function writeRuntimeBinaryFile(path, bytes) {
  if (!pyodide) throw new Error('runtime filesystem is unavailable')
  pyodide.FS.writeFile(notebookPath(path), bytes)
}
function writeRuntimeTextFile(path, text) {
  if (!pyodide) throw new Error('runtime filesystem is unavailable')
  pyodide.FS.writeFile(notebookPath(path), text)
}
async function ensureWabt() {
  if (!wabtRuntime) wabtRuntime = await loadWabt()
  return wabtRuntime
}
async function runWat2Wasm(command) {
  setRuntimeStatus(`running wat2wasm ${command.input}`)
  const wabt = await ensureWabt()
  const wat = readRuntimeTextFile(command.input)
  const module = wabt.parseWat(command.input, wat, command.features)
  try {
    module.resolveNames()
    module.validate(command.features)
    const binary = module.toBinary({ write_debug_names: true })
    writeRuntimeBinaryFile(command.output, binary.buffer)
  } finally {
    module.destroy()
  }
}
async function runWasm2Wat(command) {
  setRuntimeStatus(`running wasm2wat ${command.input}`)
  const wabt = await ensureWabt()
  const wasm = readRuntimeBinaryFile(command.input)
  const module = wabt.readWasm(new Uint8Array(wasm.bytes), {
    readDebugNames: true,
    ...command.features,
  })
  try {
    module.generateNames()
    module.applyNames()
    const text = module.toText(command.textOptions)
    if (command.output) {
      writeRuntimeTextFile(command.output, text)
    } else {
      emitOutput({ type: 'stream', name: 'stdout', text: text.endsWith('\n') ? text : `${text}\n` })
    }
  } finally {
    module.destroy()
  }
}
async function runNotebookShellCommands(commands) {
  for (const command of commands) {
    if (command.type === 'wat2wasm') {
      await runWat2Wasm(command)
    } else if (command.type === 'wasm2wat') {
      await runWasm2Wat(command)
    }
  }
}
function unsupportedReason(code) {
  const writeFile = writeFileDirective(code)
  if (writeFile) return writeFile.error
  for (const line of code.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const directive = pipInstallDirective(trimmed)
    if (directive) {
      if (directive.error) return directive.error
      continue
    }
    const runtimeDirective = browserRuntimeDirective(trimmed)
    if (runtimeDirective) {
      if (runtimeDirective.error) return runtimeDirective.error
      continue
    }
    const shell = shellDirective(trimmed)
    if (shell) {
      if (shell.error) return shell.error
      continue
    }
    if (trimmed.startsWith('%timeit')) continue
    if (trimmed.startsWith('%%')) return 'cell magics are unavailable in the browser runtime'
    if (trimmed.startsWith('%')) return 'IPython magics are unavailable in the browser runtime'
    if (trimmed.startsWith('!')) return 'shell escapes are unavailable in the browser runtime'
  }
}
function emitDirectiveWarnings(warnings) {
  for (const warning of warnings) {
    if (!warning) continue
    emitOutput({ type: 'stream', name: 'stderr', text: warning + '\n' })
  }
}
async function ensurePyodide(indexURL) {
  if (pyodide) return pyodide
  setRuntimeStatus('loading pyodide')
  const pyodideModule = await import(indexURL.replace(/\/?$/, '/') + 'pyodide.mjs')
  const loadPyodideRuntime = pyodideModule.loadPyodide
  if (typeof loadPyodideRuntime !== 'function') throw new Error('loadPyodide was not installed')
  pyodide = await loadPyodideRuntime({ indexURL })
  setRuntimeStatus('initializing python')
  pyodide.setStdout({
    write: bytes => bufferStreamBytesForCell(currentCellId, 'stdout', bytes, stdoutDecoder),
  })
  pyodide.setStderr({
    write: bytes => bufferStreamBytesForCell(currentCellId, 'stderr', bytes, stderrDecoder),
  })
  globalThis.quartz_notebook_display = handleDisplayPayload
  globalThis.quartz_notebook_fetch = input => sandboxFetch(input, currentCellId)
  globalThis.quartz_notebook_python_error = handlePythonError
  installNotebookMlBridge(globalThis)
  pyodide.runPython(notebookRuntimeBootstrap)
  return pyodide
}
async function runCell(message) {
  currentCellId = message.cellId
  debugEnabled = message.debug === true
  lastPythonError = undefined
  const reason = unsupportedReason(message.code)
  if (reason) {
    emitOutput({
      type: 'error',
      ename: 'UnsupportedRuntimeFeature',
      evalue: reason,
      traceback: reason,
    })
    post({ type: 'done', cellId: message.cellId })
    currentCellId = ''
    return
  }
  let phase = 'preparing cell'
  try {
    const prepared = stripPackageDirectives(message.code)
    prepared.code = translateLineMagics(prepared.code)
    const shellCommands = [...prepared.shellCommands]
    const directiveWarnings = [...prepared.warnings]
    phase = 'loading pyodide'
    const runtime = await ensurePyodide(message.pyodideIndexUrl)
    const modules = []
    const moduleRequirements = []
    if (Array.isArray(message.modules)) {
      phase = 'registering notebook imports'
      const register = runtime.globals.get('__quartz_register_notebook_module')
      if (!register || typeof register !== 'function')
        throw new Error('runtime notebook importer is unavailable')
      for (const module of message.modules) {
        if (
          !module ||
          typeof module.name !== 'string' ||
          typeof module.source !== 'string' ||
          typeof module.sourcePath !== 'string'
        )
          continue
        const preparedModule = stripPackageDirectives(module.source)
        preparedModule.code = translateLineMagics(preparedModule.code)
        moduleRequirements.push(...preparedModule.requirements)
        shellCommands.push(...preparedModule.shellCommands)
        directiveWarnings.push(...preparedModule.warnings)
        const runtimeModule = {
          name: module.name,
          source: preparedModule.code,
          sourcePath: module.sourcePath,
        }
        modules.push(runtimeModule)
        register(runtimeModule.name, runtimeModule.source, runtimeModule.sourcePath)
      }
      if (register && 'destroy' in register && typeof register.destroy === 'function') {
        register.destroy()
      }
    }
    emitDirectiveWarnings(directiveWarnings)
    await runNotebookShellCommands(shellCommands)
    if (usesNotebookMlRuntime(prepared.code, modules)) {
      phase = 'loading notebook ml runtime'
      setRuntimeStatus('loading webgpu')
      await ensureNotebookMlRuntime()
    }
    phase = 'loading packages'
    await preparePackages(
      runtime,
      prepared.code,
      prepared.requirements.concat(moduleRequirements),
      modules,
    )
    phase = 'running python'
    const runner = runtime.globals.get('__quartz_run_cell')
    if (!runner || typeof runner !== 'function')
      throw new Error('runtime cell runner is unavailable')
    const result = await runner(prepared.code)
    if (
      result &&
      typeof result === 'object' &&
      'destroy' in result &&
      typeof result.destroy === 'function'
    ) {
      result.destroy()
    }
  } catch (error) {
    flushStreamsForCell(message.cellId)
    if (phase === 'running python') {
      emitPythonError(error, phase)
    } else {
      emitError(error, phase)
    }
  } finally {
    flushStreamsForCell(message.cellId)
    post({ type: 'done', cellId: message.cellId })
    currentCellId = ''
    debugEnabled = false
    lastPythonError = undefined
  }
}
globalThis.addEventListener('message', event => {
  const message = event.data
  if (!message || message.source !== source) return
  if (message.type === 'init') {
    runtimeId = message.runtimeId
    post({ type: 'ready' })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    runCell(message)
  } else if (message.type === 'asset-result' && message.runtimeId === runtimeId) {
    const pending = pendingAssets.get(message.assetId)
    if (!pending) return
    pendingAssets.delete(message.assetId)
    if (!message.ok) {
      pending.reject(new Error(message.error || 'failed to fetch notebook asset'))
      return
    }
    pending.resolve(
      new Response(message.bytes, {
        status: message.status,
        statusText: message.statusText,
        headers: { 'content-type': message.contentType || 'application/octet-stream' },
      }),
    )
  } else if (message.type === 'file' && message.runtimeId === runtimeId) {
    try {
      const file = readRuntimeBinaryFile(message.path)
      post(
        {
          type: 'file-result',
          requestId: message.requestId,
          ok: true,
          status: 200,
          statusText: 'OK',
          contentType: file.contentType,
          bytes: file.bytes,
        },
        [file.bytes],
      )
    } catch (error) {
      post({
        type: 'file-result',
        requestId: message.requestId,
        ok: false,
        status: 404,
        statusText: 'Not Found',
        contentType: 'text/plain',
        error: textOf(error),
      })
    }
  }
})
