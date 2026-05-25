const source = 'quartz-notebook-runtime'
const decoder = new TextDecoder()
let runtimeId = ''
let currentCellId = ''
let goEval

function post(message, transfer) {
  globalThis.postMessage({ source, runtimeId, ...message }, transfer ?? [])
}

function textOf(value) {
  return value instanceof Error ? value.message : String(value)
}

function emitOutput(cellId, output) {
  post({ type: 'output', cellId, output })
}

function emitStream(text) {
  if (!currentCellId || text.length === 0) return
  emitOutput(currentCellId, { type: 'stream', name: 'stdout', text })
}

function emitError(cellId, error) {
  const text = textOf(error)
  emitOutput(cellId, { type: 'error', ename: 'GoError', evalue: text, traceback: text })
}

function installOutputCapture() {
  const fs = globalThis.fs
  if (!fs || typeof fs.writeSync !== 'function') return
  fs.writeSync = (_fd, bytes) => {
    emitStream(decoder.decode(bytes))
    return bytes.length
  }
}

async function loadScript(url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Go runtime script request failed with ${response.status}`)
  const code = await response.text()
  globalThis.Function(code)()
}

function isObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readChunkedAssetManifest(value) {
  if (!isObject(value) || value.version !== 1) return
  if (!Number.isSafeInteger(value.size) || value.size < 0) return
  if (!Array.isArray(value.chunks)) return
  if (!value.chunks.every(chunk => typeof chunk === 'string' && chunk.length > 0)) return
  return { size: value.size, chunks: value.chunks }
}

function isChunkedAssetResponse(url, response) {
  const contentType = response.headers.get('content-type') ?? ''
  const responseUrl = response.url || url
  return contentType.includes('json') || responseUrl.includes('.chunks') || url.includes('.chunks')
}

function concatChunks(chunks, size, label) {
  const actualSize = chunks.reduce((total, chunk) => total + chunk.byteLength, 0)
  if (actualSize !== size) throw new Error(`Go ${label} chunk size mismatch`)
  const bytes = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return bytes
}

async function fetchChunkedBytes(url, manifest, label) {
  const chunks = await Promise.all(
    manifest.chunks.map(async chunk => {
      const chunkUrl = new URL(chunk, url).href
      const response = await fetch(chunkUrl)
      if (!response.ok) throw new Error(`Go ${label} chunk request failed with ${response.status}`)
      return new Uint8Array(await response.arrayBuffer())
    }),
  )
  return concatChunks(chunks, manifest.size, label)
}

async function fetchBytes(url, label) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Go ${label} request failed with ${response.status}`)
  if (!isChunkedAssetResponse(url, response)) return new Uint8Array(await response.arrayBuffer())
  const manifest = readChunkedAssetManifest(await response.json())
  if (!manifest) throw new Error(`Go ${label} chunk manifest is invalid`)
  return await fetchChunkedBytes(url, manifest, label)
}

async function instantiateGo(wasmUrl, go) {
  const bytes = await fetchBytes(wasmUrl, 'runtime wasm')
  return await WebAssembly.instantiate(bytes, go.importObject)
}

function assetEnding(assets, ending) {
  const asset = assets.find(item => item.includes(ending))
  if (!asset) throw new Error(`Go runtime asset is missing ${ending}`)
  return asset
}

async function waitForEval() {
  for (let tries = 0; tries < 100; tries += 1) {
    if (typeof globalThis.quartzNativeGoEval === 'function') {
      goEval = globalThis.quartzNativeGoEval
      return
    }
    await new Promise(resolve => globalThis.setTimeout(resolve, 10))
  }
  throw new Error('Go runtime did not expose quartzNativeGoEval')
}

async function init(message) {
  const assets = Array.isArray(message.assets)
    ? message.assets.filter(item => typeof item === 'string')
    : []
  const wasmExecUrl = assetEnding(assets, '/wasm_exec')
  const wasmUrl = assetEnding(assets, '/yaegi')
  await loadScript(wasmExecUrl)
  installOutputCapture()
  const Go = globalThis.Go
  if (typeof Go !== 'function') throw new Error('wasm_exec.js did not expose Go')
  const go = new Go()
  const result = await instantiateGo(wasmUrl, go)
  void go.run(result.instance).catch(error => {
    post({ type: 'status', text: `Go runtime stopped: ${textOf(error)}` })
  })
  await waitForEval()
  post({ type: 'ready' })
}

function run(message) {
  const cellId = typeof message.cellId === 'string' ? message.cellId : ''
  const code = typeof message.code === 'string' ? message.code : ''
  if (!cellId) return
  currentCellId = cellId
  let failed = false
  try {
    if (typeof goEval !== 'function') throw new Error('Go runtime is not ready')
    const result = JSON.parse(String(goEval(code)))
    if (result.ok !== true) {
      failed = true
      emitError(cellId, typeof result.error === 'string' ? result.error : 'Go evaluation failed')
    } else if (typeof result.value === 'string' && result.value.length > 0) {
      emitOutput(cellId, { type: 'text', text: result.value })
    }
  } catch (error) {
    failed = true
    emitError(cellId, error)
  } finally {
    currentCellId = ''
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
      post({ type: 'status', text: `Go runtime init failed: ${textOf(error)}` })
      throw error
    })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    run(message)
  }
})
