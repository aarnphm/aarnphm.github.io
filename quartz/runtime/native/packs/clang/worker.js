const source = 'quartz-notebook-runtime'
const decoder = new TextDecoder()
const clangAssetNames = [
  'llvm.core.wasm',
  'llvm.core2.wasm',
  'llvm.core3.wasm',
  'llvm.core4.wasm',
  'llvm-resources.tar',
]

let runtimeId = ''
let language = 'c'
let runClang
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

function emitError(cellId, name, error) {
  const text = textOf(error)
  emitOutput(cellId, { type: 'error', ename: name, evalue: text, traceback: text })
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
  if (actualSize !== size) throw new Error(`Clang ${label} chunk size mismatch`)
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
    manifest.chunks.map(async (chunk, index) => {
      const chunkUrl = new URL(chunk, url).href
      post({
        type: 'status',
        text: `fetching Clang ${label} chunk ${index + 1}/${manifest.chunks.length}`,
      })
      const response = await fetch(chunkUrl)
      if (!response.ok) throw new Error(`Clang ${label} chunk request failed with ${response.status}`)
      const bytes = new Uint8Array(await response.arrayBuffer())
      post({
        type: 'status',
        text: `loaded Clang ${label} chunk ${index + 1}/${manifest.chunks.length}`,
      })
      return bytes
    }),
  )
  return concatChunks(chunks, manifest.size, label)
}

async function fetchText(url, label) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Clang ${label} request failed with ${response.status}`)
  return await response.text()
}

function assetFor(assets, filename) {
  const asset = assets.find(item => item.includes(filename))
  if (!asset) throw new Error(`Clang runtime asset is missing ${filename}`)
  return asset
}

function assetUrl(value) {
  const base =
    typeof globalThis.location?.href === 'string' ? globalThis.location.href : 'http://localhost/'
  return new URL(value, base).href
}

function assetContentType(filename) {
  return filename.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream'
}

function assetHeaders(response, filename, byteLength) {
  const headers = new Headers(response?.headers)
  headers.set('content-type', assetContentType(filename))
  if (byteLength !== undefined) {
    headers.set('content-length', String(byteLength))
  } else if (!headers.has('content-length')) {
    headers.delete('content-length')
  }
  return headers
}

function directAssetResponse(response, filename) {
  const length = response.headers.get('content-length')
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: assetHeaders(response, filename, length ?? undefined),
  })
}

function byteAssetResponse(bytes, filename) {
  return new Response(bytes, {
    headers: assetHeaders(undefined, filename, bytes.byteLength),
  })
}

async function assetResponse(url, filename) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Clang ${filename} request failed with ${response.status}`)
  if (!isChunkedAssetResponse(url, response)) {
    post({ type: 'status', text: `fetching Clang ${filename}` })
    if (response.body) return directAssetResponse(response, filename)
    const bytes = new Uint8Array(await response.arrayBuffer())
    return byteAssetResponse(bytes, filename)
  }
  const manifest = readChunkedAssetManifest(await response.json())
  if (!manifest) throw new Error(`Clang ${filename} chunk manifest is invalid`)
  const bytes = await fetchChunkedBytes(url, manifest, filename)
  post({ type: 'status', text: `loaded Clang ${filename}` })
  return byteAssetResponse(bytes, filename)
}

function installClangAssetFetch(assetMap) {
  globalThis.quartzNativeClangFetch = async (input, init) => {
    const url =
      input instanceof URL
        ? input.href
        : typeof input === 'string'
          ? assetUrl(input)
          : typeof input?.url === 'string'
            ? assetUrl(input.url)
            : ''
    const filename = assetMap.get(url)
    if (filename) return await assetResponse(url, filename)
    return await fetch(input, init)
  }
}

function clangAssetMap(assets) {
  return new Map(clangAssetNames.map(filename => [assetUrl(assetFor(assets, filename)), filename]))
}

function rewriteClangAssetUrls(sourceText, assetMap) {
  let next = sourceText.replace(
    'var fetch_default = fetch2;',
    'var fetch_default = globalThis.quartzNativeClangFetch ?? fetch2;',
  )
  for (const [url, filename] of assetMap) {
    next = next.replaceAll(JSON.stringify(`./${filename}`), JSON.stringify(url))
  }
  return next
}

async function importClangModule(assets) {
  const assetMap = clangAssetMap(assets)
  installClangAssetFetch(assetMap)
  const sourceText = await fetchText(assetFor(assets, 'bundle.js'), 'bundle module')
  const rewritten = rewriteClangAssetUrls(sourceText, assetMap)
  return await import(`data:text/javascript;charset=utf-8,${encodeURIComponent(rewritten)}`)
}

function captureBytes() {
  const chunks = []
  return {
    write(bytes) {
      if (!(bytes instanceof Uint8Array)) return
      chunks.push(bytes.slice())
    },
    text() {
      return chunks.map(chunk => decoder.decode(chunk)).join('')
    },
  }
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

function cProgram(code) {
  const body = code.trim()
  if (/\bmain\s*\(/.test(body)) return `${body}\n`
  return ['#include <stdio.h>', 'int main(void) {', body, 'return 0;', '}', ''].join('\n')
}

function cppProgram(code) {
  const body = code.trim()
  if (/\bmain\s*\(/.test(body)) return `${body}\n`
  return ['#include <iostream>', 'int main() {', body, 'return 0;', '}', ''].join('\n')
}

function runtimeProgram(code) {
  return language === 'cpp' ? cppProgram(code) : cProgram(code)
}

function runtimeCompiler() {
  return language === 'cpp' ? 'clang++' : 'clang'
}

function runtimeCompilerArgs(filename) {
  const args = [runtimeCompiler(), filename, '-o', 'main.wasm']
  if (language === 'cpp') {
    args.push('-fwasm-exceptions', '-mllvm', '-wasm-use-legacy-eh=false', '-lc++abi', '-lunwind')
  }
  return args
}

function runtimeFilename() {
  return language === 'cpp' ? 'main.cpp' : 'main.c'
}

function runtimeErrorName() {
  return language === 'cpp' ? 'C++Error' : 'CError'
}

function requireClang() {
  if (!runClang) throw new Error('Clang runtime is not ready')
  return runClang
}

function requireWasiShim() {
  if (!wasiShim) throw new Error('Clang WASI shim is not ready')
  return wasiShim
}

async function compileProgram(code) {
  const clang = requireClang()
  const filename = runtimeFilename()
  const stdout = captureBytes()
  const stderr = captureBytes()
  post({ type: 'status', text: `compiling ${language === 'cpp' ? 'C++' : 'C'} with Clang` })
  try {
    const files = await clang(
      runtimeCompilerArgs(filename),
      { [filename]: runtimeProgram(code) },
      {
        stdout: stdout.write,
        stderr: stderr.write,
        decodeASCII: false,
        fetchProgress: () => {},
      },
    )
    return { wasm: files['main.wasm'], stdout: stdout.text(), stderr: stderr.text() }
  } catch (error) {
    if (error && typeof error === 'object') {
      error.stdout = stdout.text()
      error.stderr = stderr.text()
    }
    throw error
  }
}

async function runWasiProgram(wasm) {
  const { Directory, Fd, PreopenDirectory, WASI } = requireWasiShim()
  post({ type: 'status', text: `running ${language === 'cpp' ? 'C++' : 'C'} wasm` })
  const stdin = captureStdin(Fd)
  const stdout = captureStdio(Fd)
  const stderr = captureStdio(Fd)
  const root = new PreopenDirectory('/', [['tmp', new Directory([])]])
  const wasi = new WASI(['main.wasm'], [], [stdin, stdout, stderr, root], { debug: false })
  const { instance } = await WebAssembly.instantiate(wasm, {
    wasi_snapshot_preview1: wasi.wasiImport,
  })
  try {
    const exitCode = wasi.start(instance)
    return { exitCode, stdout, stderr }
  } catch (error) {
    if (typeof WebAssembly.Exception === 'function' && error instanceof WebAssembly.Exception) {
      return { exitCode: 1, stdout, stderr, runtimeError: 'terminating: uncaught C++ exception' }
    }
    throw error
  }
}

async function init(message) {
  language = message.language === 'cpp' ? 'cpp' : 'c'
  const assets = Array.isArray(message.assets)
    ? message.assets.filter(item => typeof item === 'string')
    : []
  post({ type: 'status', text: `loading ${language === 'cpp' ? 'C++' : 'C'} runtime pack` })
  const [clangModule, nextWasiShim] = await Promise.all([
    importClangModule(assets),
    import(assetFor(assets, 'browser-wasi-shim.mjs')),
  ])
  runClang = clangModule.runClang
  wasiShim = nextWasiShim
  post({ type: 'status', text: 'initializing Clang compiler' })
  await requireClang()(null, {}, {
    fetchProgress: progress => {
      if (!progress || !Number.isFinite(progress.totalLength) || progress.totalLength <= 0) return
      const percent = Math.floor((progress.doneLength / progress.totalLength) * 100)
      post({ type: 'status', text: `loaded Clang ${percent}%` })
    },
  })
  post({ type: 'ready' })
}

async function run(message) {
  const cellId = typeof message.cellId === 'string' ? message.cellId : ''
  const code = typeof message.code === 'string' ? message.code : ''
  if (!cellId) return
  let failed = false
  try {
    const compiled = await compileProgram(code)
    if (compiled.stdout.length > 0) {
      emitOutput(cellId, { type: 'stream', name: 'stdout', text: compiled.stdout })
    }
    if (compiled.stderr.length > 0) {
      emitOutput(cellId, { type: 'stream', name: 'stderr', text: compiled.stderr })
    }
    if (!(compiled.wasm instanceof Uint8Array)) throw new Error('Clang did not emit main.wasm')
    const result = await runWasiProgram(compiled.wasm)
    emitCaptured(cellId, 'stdout', result.stdout)
    emitCaptured(cellId, 'stderr', result.stderr)
    if (result.runtimeError) emitError(cellId, runtimeErrorName(), result.runtimeError)
    failed = Boolean(result.runtimeError) || result.exitCode !== 0 || result.stderr.text().length > 0
  } catch (error) {
    failed = true
    if (error && typeof error === 'object') {
      if (typeof error.stdout === 'string' && error.stdout.length > 0) {
        emitOutput(cellId, { type: 'stream', name: 'stdout', text: error.stdout })
      }
      if (typeof error.stderr === 'string' && error.stderr.length > 0) {
        emitOutput(cellId, { type: 'stream', name: 'stderr', text: error.stderr })
      }
    }
    emitError(cellId, runtimeErrorName(), error)
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
      post({ type: 'status', text: `Clang runtime init failed: ${textOf(error)}` })
      throw error
    })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    void run(message)
  }
})
