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
function stripPackageDirectives(code) {
  const lines = []
  const requirements = []
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
    lines.push(line)
  }
  return { code: lines.join('\n'), requirements, warnings }
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
function unsupportedReason(code) {
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
  }
})
