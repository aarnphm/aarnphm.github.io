const source = 'quartz-notebook-runtime'
let runtimeId = ''
let currentCellId = ''
let mainFunc
let stderrText = ''

const topLevelMainPattern = /^main(?:[ \t]*::|[ \t]*(?:[^\s=][^=\n]*)?=)/m

function post(message) {
  globalThis.postMessage({ source, runtimeId, ...message })
}

function textOf(value) {
  return value instanceof Error ? value.message : String(value)
}

function emitOutput(cellId, output) {
  post({ type: 'output', cellId, output })
}

function emitStream(name, text) {
  if (!currentCellId || text.length === 0) return
  if (name === 'stderr') stderrText += text
  emitOutput(currentCellId, { type: 'stream', name, text })
}

function emitError(cellId, error) {
  const text = textOf(error)
  emitOutput(cellId, { type: 'error', ename: 'HaskellError', evalue: text, traceback: text })
}

export function haskellNotebookSource(code) {
  if (topLevelMainPattern.test(code)) return code
  const source = code.trimEnd()
  const separator = source.length > 0 ? '\n\n' : ''
  return `${source}${separator}main :: IO ()\nmain = pure ()\n`
}

function assetToken(assets, token) {
  const asset = assets.find(item => item.includes(token))
  if (!asset) throw new Error(`Haskell runtime asset is missing ${token}`)
  return asset
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
  const contentType = response?.headers?.get?.('content-type') ?? ''
  const responseUrl = response?.url || url
  return (
    url.endsWith('.chunks.json') ||
    contentType.includes('json') ||
    responseUrl.includes('.chunks') ||
    url.includes('.chunks')
  )
}

function concatChunks(chunks, size, label) {
  const actualSize = chunks.reduce((total, chunk) => total + chunk.byteLength, 0)
  if (actualSize !== size) throw new Error(`Haskell ${label} chunk size mismatch`)
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
      if (!response.ok) {
        throw new Error(`Haskell ${label} chunk request failed with ${response.status}`)
      }
      return new Uint8Array(await response.arrayBuffer())
    }),
  )
  return concatChunks(chunks, manifest.size, label)
}

async function fetchBytes(url, label = url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Haskell ${label} request failed with ${response.status}`)
  if (isChunkedAssetResponse(url, response)) {
    const manifest = readChunkedAssetManifest(await response.json())
    if (!manifest) throw new Error(`Haskell ${label} chunk manifest is invalid`)
    return await fetchChunkedBytes(url, manifest, label)
  }
  return new Uint8Array(await response.arrayBuffer())
}

function createChunkedStream(url, manifest, label) {
  let chunkIndex = 0
  let activeReader = null
  let actualSize = 0
  return new ReadableStream({
    async pull(controller) {
      while (true) {
        if (activeReader) {
          const { done, value } = await activeReader.read()
          if (!done) {
            if (value && value.byteLength > 0) {
              actualSize += value.byteLength
              controller.enqueue(value)
              return
            }
            continue
          }
          activeReader = null
        }
        if (chunkIndex >= manifest.chunks.length) {
          if (actualSize !== manifest.size) {
            throw new Error(`Haskell ${label} chunk size mismatch`)
          }
          controller.close()
          return
        }
        const chunk = manifest.chunks[chunkIndex]
        chunkIndex += 1
        const chunkUrl = new URL(chunk, url).href
        const chunkResponse = await fetch(chunkUrl)
        if (!chunkResponse.ok) {
          throw new Error(`Haskell ${label} chunk request failed with ${chunkResponse.status}`)
        }
        if (chunkResponse.body && typeof chunkResponse.body.getReader === 'function') {
          activeReader = chunkResponse.body.getReader()
          continue
        }
        const bytes = new Uint8Array(await chunkResponse.arrayBuffer())
        actualSize += bytes.byteLength
        if (bytes.byteLength > 0) {
          controller.enqueue(bytes)
          return
        }
      }
    },
    async cancel(reason) {
      if (activeReader) {
        try {
          await activeReader.cancel(reason)
        } catch {}
        activeReader = null
      }
    },
  })
}

async function fetchStream(url, label = url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Haskell ${label} request failed with ${response.status}`)
  if (!isChunkedAssetResponse(url, response)) {
    if (response.body) return response.body
    const bytes = new Uint8Array(await response.arrayBuffer())
    return new ReadableStream({
      start(controller) {
        if (bytes.byteLength > 0) controller.enqueue(bytes)
        controller.close()
      },
    })
  }
  const manifest = readChunkedAssetManifest(await response.json())
  if (!manifest) throw new Error(`Haskell ${label} chunk manifest is invalid`)
  return createChunkedStream(url, manifest, label)
}

async function compileWasm(url, label = 'bsdtar wasm') {
  if (
    typeof WebAssembly.compileStreaming === 'function' &&
    typeof ReadableStream === 'function'
  ) {
    try {
      const stream = await fetchStream(url, label)
      const response = new Response(stream, {
        headers: { 'content-type': 'application/wasm' },
      })
      return await WebAssembly.compileStreaming(response)
    } catch (error) {
      if (error instanceof Error && error.message.startsWith('Haskell ')) throw error
    }
  }
  return await WebAssembly.compile(await fetchBytes(url, label))
}

async function fetchText(url, label) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Haskell ${label} request failed with ${response.status}`)
  return await response.text()
}

async function instantiateWasi(url, imports) {
  const module = await compileWasm(url, 'bsdtar wasm')
  const instantiated = await WebAssembly.instantiate(module, imports)
  const instance =
    instantiated instanceof WebAssembly.Instance || !('instance' in instantiated)
      ? instantiated
      : instantiated.instance
  return { module, instance }
}

async function importRewrittenModule(rewritten) {
  if (typeof Blob === 'function' && typeof URL?.createObjectURL === 'function') {
    const objectUrl = URL.createObjectURL(new Blob([rewritten], { type: 'text/javascript' }))
    try {
      return await import(objectUrl)
    } finally {
      if (typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(objectUrl)
    }
  }
  return await import(`data:text/javascript;charset=utf-8,${encodeURIComponent(rewritten)}`)
}

async function importDyldModule(assets) {
  const dyldUrl = assetToken(assets, '/dyld')
  const preludeUrl = assetToken(assets, '/prelude')
  const postLinkUrl = assetToken(assets, '/post-link')
  const wasiShimUrl = assetToken(assets, '/browser-wasi-shim')
  const sourceText = (await fetchText(dyldUrl, 'dyld module'))
    .replaceAll('"./prelude.mjs"', JSON.stringify(preludeUrl))
    .replaceAll('"./post-link.mjs"', JSON.stringify(postLinkUrl))
    .replaceAll('"https://esm.sh/gh/haskell-wasm/browser_wasi_shim"', JSON.stringify(wasiShimUrl))
  return await importRewrittenModule(sourceText)
}

async function extractRootfs(assets, wasiShim) {
  const { ConsoleStdout, File, OpenFile, PreopenDirectory, WASI } = wasiShim
  const rootfs = new PreopenDirectory('/', [])
  const bsdtarWasi = new WASI(
    ['bsdtar.wasm', '-x'],
    [],
    [
      new OpenFile(new File(new Uint8Array(), { readonly: true })),
      ConsoleStdout.lineBuffered(msg => post({ type: 'status', text: msg })),
      ConsoleStdout.lineBuffered(msg => post({ type: 'status', text: msg })),
      rootfs,
    ],
    { debug: false },
  )
  const bsdtarUrl = assetToken(assets, '/bsdtar')
  const rootfsUrl = assetToken(assets, '/rootfs.tar')
  const [{ instance }, rootfsBytes] = await Promise.all([
    instantiateWasi(bsdtarUrl, { wasi_snapshot_preview1: bsdtarWasi.wasiImport }),
    fetchBytes(rootfsUrl, 'rootfs tarball'),
  ])
  bsdtarWasi.fds[0] = new OpenFile(new File(rootfsBytes, { readonly: true }))
  const exitCode = bsdtarWasi.start(instance)
  if (exitCode !== 0) throw new Error(`Haskell rootfs extraction failed with ${exitCode}`)
  return rootfs
}

async function init(message) {
  const assets = Array.isArray(message.assets)
    ? message.assets.filter(item => typeof item === 'string')
    : []
  post({ type: 'status', text: 'loading Haskell runtime pack' })
  const wasiShim = await import(assetToken(assets, '/browser-wasi-shim'))
  const [{ DyLDBrowserHost, main }, rootfs] = await Promise.all([
    importDyldModule(assets),
    extractRootfs(assets, wasiShim),
  ])
  post({ type: 'status', text: 'linking Haskell runtime pack' })
  const dyld = await main({
    rpc: new DyLDBrowserHost({
      rootfs,
      stdout: msg => emitStream('stdout', `${msg}\n`),
      stderr: msg => emitStream('stderr', `${msg}\n`),
    }),
    searchDirs: ['/tmp/clib', '/tmp/hslib/lib/wasm32-wasi-ghc-9.14.0.20251031-inplace'],
    mainSoPath: '/tmp/libplayground001.so',
    args: ['libplayground001.so', '+RTS', '-c', '-RTS'],
    isIserv: false,
  })
  mainFunc = await dyld.exportFuncs.myMain('/tmp/hslib/lib')
  post({ type: 'ready' })
}

async function run(message) {
  const cellId = typeof message.cellId === 'string' ? message.cellId : ''
  const code = typeof message.code === 'string' ? message.code : ''
  if (!cellId) return
  currentCellId = cellId
  stderrText = ''
  let failed = false
  try {
    if (typeof mainFunc !== 'function') throw new Error('Haskell runtime is not ready')
    await mainFunc('', haskellNotebookSource(code))
    if (stderrText.length > 0) failed = true
  } catch (error) {
    failed = true
    emitError(cellId, error)
  } finally {
    currentCellId = ''
    post({ type: 'done', cellId, failed })
  }
}

function handleMessage(event) {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== globalThis.location.origin) return
  const message = event.data
  if (!message || typeof message !== 'object' || message.source !== source) return
  if (message.type === 'init') {
    const nextRuntimeId = typeof message.runtimeId === 'string' ? message.runtimeId : ''
    if (!nextRuntimeId) return
    runtimeId = nextRuntimeId
    void init(message).catch(error => {
      post({ type: 'status', text: `Haskell runtime init failed: ${textOf(error)}` })
      throw error
    })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    void run(message)
  }
}

if (typeof globalThis.addEventListener === 'function') {
  globalThis.addEventListener('message', handleMessage)
}
