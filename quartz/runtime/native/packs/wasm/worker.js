const source = 'quartz-notebook-runtime'
const decoder = new TextDecoder()

let runtimeId = ''
let loadWabt
let wabtRuntime
let wasiShim

function post(message, transfer) {
  globalThis.postMessage({ source, runtimeId, ...message }, transfer ?? [])
}

function textOf(value) {
  return value instanceof Error ? value.message : String(value)
}

function emitOutput(cellId, output) {
  post({ type: 'output', cellId, output })
}

function emitError(cellId, error) {
  const text = textOf(error)
  emitOutput(cellId, { type: 'error', ename: 'WasmError', evalue: text, traceback: text })
}

function assetFor(assets, filename) {
  const asset = assets.find(item => item.includes(filename))
  if (!asset) throw new Error(`Wasm runtime asset is missing ${filename}`)
  return asset
}

function wabtFeatures() {
  return {
    annotations: true,
    bulk_memory: true,
    exceptions: true,
    extended_const: true,
    function_references: true,
    gc: true,
    memory64: true,
    multi_memory: true,
    multi_value: true,
    mutable_globals: true,
    reference_types: true,
    relaxed_simd: true,
    sat_float_to_int: true,
    sign_extension: true,
    simd: true,
    tail_call: true,
    threads: true,
  }
}

async function ensureWabt() {
  if (!loadWabt) throw new Error('WABT loader is not ready')
  if (!wabtRuntime) wabtRuntime = await loadWabt()
  return wabtRuntime
}

function hasWasmMagic(bytes) {
  return (
    bytes.byteLength >= 4 &&
    bytes[0] === 0x00 &&
    bytes[1] === 0x61 &&
    bytes[2] === 0x73 &&
    bytes[3] === 0x6d
  )
}

function base64Bytes(value) {
  const raw = atob(value.replace(/\s/g, ''))
  const bytes = new Uint8Array(raw.length)
  for (let index = 0; index < raw.length; index += 1) bytes[index] = raw.charCodeAt(index)
  return bytes
}

function hexBytes(value) {
  const normalized = value.replace(/^0x/i, '').replace(/[\s,]/g, '')
  const bytes = new Uint8Array(normalized.length / 2)
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(normalized.slice(index * 2, index * 2 + 2), 16)
  }
  return bytes
}

async function watBytes(sourceText) {
  const wabt = await ensureWabt()
  const features = wabtFeatures()
  const module = wabt.parseWat('notebook.wat', sourceText, features)
  try {
    module.resolveNames()
    module.validate(features)
    return module.toBinary({ write_debug_names: true }).buffer
  } finally {
    module.destroy()
  }
}

async function wasmBytes(sourceText) {
  const trimmed = sourceText.trim()
  if (trimmed.startsWith('data:application/wasm;base64,')) {
    return base64Bytes(trimmed.slice('data:application/wasm;base64,'.length))
  }
  const maybeHex = trimmed.replace(/^0x/i, '').replace(/[\s,]/g, '')
  if (/^(?:[0-9a-fA-F]{2})+$/.test(maybeHex)) {
    const bytes = hexBytes(maybeHex)
    if (hasWasmMagic(bytes)) return bytes
  }
  if (/^[A-Za-z0-9+/=\s]+$/.test(trimmed)) {
    try {
      const bytes = base64Bytes(trimmed)
      if (hasWasmMagic(bytes)) return bytes
    } catch {}
  }
  return await watBytes(sourceText)
}

function captureStdio(Fd) {
  return new (class extends Fd {
    constructor() {
      super()
      this.chunks = []
    }

    text() {
      return this.chunks.map(chunk => decoder.decode(chunk)).join('')
    }

    fd_write(data) {
      this.chunks.push(data.slice())
      return { ret: 0, nwritten: data.byteLength }
    }
  })()
}

function captureStdin(Fd) {
  return new (class extends Fd {
    fd_read() {
      return { ret: 0, data: new Uint8Array() }
    }
  })()
}

function emitCaptured(cellId, name, stdio) {
  const text = stdio.text()
  if (text.length > 0) emitOutput(cellId, { type: 'stream', name, text })
}

function defaultTable() {
  try {
    return new WebAssembly.Table({ initial: 1, element: 'funcref' })
  } catch {
    return new WebAssembly.Table({ initial: 1, element: 'anyfunc' })
  }
}

function fallbackImport(cellId, item) {
  if (item.kind === 'function') {
    return (...args) => {
      emitOutput(cellId, {
        type: 'stream',
        name: 'stdout',
        text: `${item.module}.${item.name}(${args.map(String).join(', ')})\n`,
      })
      return 0
    }
  }
  if (item.kind === 'memory') return new WebAssembly.Memory({ initial: 1 })
  if (item.kind === 'table') return defaultTable()
  if (item.kind === 'global') return new WebAssembly.Global({ value: 'i32', mutable: true }, 0)
}

function ensureImportObject(importObject, moduleName) {
  const entry = importObject[moduleName]
  if (entry && typeof entry === 'object') return entry
  const next = {}
  importObject[moduleName] = next
  return next
}

function buildImportObject(module, cellId, wasi) {
  const importObject = {}
  if (wasi) importObject.wasi_snapshot_preview1 = wasi.wasiImport
  for (const item of WebAssembly.Module.imports(module)) {
    const moduleImports = ensureImportObject(importObject, item.module)
    if (item.name in moduleImports) continue
    moduleImports[item.name] = fallbackImport(cellId, item)
  }
  return importObject
}

function callableExportName(module, instance) {
  const exportedFunctions = WebAssembly.Module.exports(module)
    .filter(item => item.kind === 'function')
    .map(item => item.name)
    .filter(name => typeof instance.exports[name] === 'function')
  for (const name of ['main', 'run', 'start']) {
    if (exportedFunctions.includes(name)) return name
  }
  return exportedFunctions.length === 1 ? exportedFunctions[0] : undefined
}

function resultText(value) {
  if (value === undefined) return undefined
  return typeof value === 'bigint' ? `${value}n` : String(value)
}

function exportList(module) {
  const names = WebAssembly.Module.exports(module).map(item => item.name)
  return names.length > 0 ? names.join(', ') : 'none'
}

async function instantiateAndRun(cellId, bytes) {
  if (!wasiShim) throw new Error('Wasm WASI shim is not ready')
  const module = await WebAssembly.compile(bytes)
  const { Directory, Fd, PreopenDirectory, WASI } = wasiShim
  const stdin = captureStdin(Fd)
  const stdout = captureStdio(Fd)
  const stderr = captureStdio(Fd)
  const root = new PreopenDirectory('/', [['tmp', new Directory([])]])
  const needsWasi = WebAssembly.Module.imports(module).some(
    item => item.module === 'wasi_snapshot_preview1',
  )
  const wasi = needsWasi ? new WASI(['notebook.wasm'], [], [stdin, stdout, stderr, root], { debug: false }) : undefined
  const instance = await WebAssembly.instantiate(module, buildImportObject(module, cellId, wasi))
  let failed = false
  if (wasi && typeof instance.exports._start === 'function') {
    failed = wasi.start(instance) !== 0
  } else if (typeof instance.exports._start === 'function') {
    instance.exports._start()
  } else {
    const name = callableExportName(module, instance)
    if (name) {
      const text = resultText(instance.exports[name]())
      if (text !== undefined) emitOutput(cellId, { type: 'text', text: `${name}() = ${text}` })
    } else {
      emitOutput(cellId, { type: 'text', text: `exports: ${exportList(module)}` })
    }
  }
  emitCaptured(cellId, 'stdout', stdout)
  emitCaptured(cellId, 'stderr', stderr)
  return failed || stderr.text().length > 0
}

async function init(message) {
  const assets = Array.isArray(message.assets)
    ? message.assets.filter(item => typeof item === 'string')
    : []
  post({ type: 'status', text: 'loading Wasm runtime pack' })
  const [wabtModule, nextWasiShim] = await Promise.all([
    import(assetFor(assets, 'wabt.mjs')),
    import(assetFor(assets, 'browser-wasi-shim.mjs')),
  ])
  loadWabt = wabtModule.default
  wasiShim = nextWasiShim
  await ensureWabt()
  post({ type: 'ready' })
}

async function run(message) {
  const cellId = typeof message.cellId === 'string' ? message.cellId : ''
  const code = typeof message.code === 'string' ? message.code : ''
  if (!cellId) return
  let failed = false
  try {
    failed = await instantiateAndRun(cellId, await wasmBytes(code))
  } catch (error) {
    failed = true
    emitError(cellId, error)
  } finally {
    post({ type: 'done', cellId, failed })
  }
}

globalThis.addEventListener('message', event => {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== globalThis.location.origin) return
  const message = event.data
  if (!message || typeof message !== 'object' || message.source !== source) return
  if (message.type === 'init') {
    const nextRuntimeId = typeof message.runtimeId === 'string' ? message.runtimeId : ''
    if (!nextRuntimeId) return
    runtimeId = nextRuntimeId
    void init(message).catch(error => {
      post({ type: 'status', text: `Wasm runtime init failed: ${textOf(error)}` })
      throw error
    })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    void run(message)
  }
})
