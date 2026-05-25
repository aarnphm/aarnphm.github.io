const source = 'quartz-notebook-runtime'
const encoder = new TextEncoder()
const decoder = new TextDecoder()

const sysrootFiles = [
  'libaddr2line-b8754aeb03c02354.rlib.gz',
  'libadler-05c3545f6cd12159.rlib.gz',
  'liballoc-0dab879bc41cd6bd.rlib.gz',
  'libcfg_if-c7fd2cef50341546.rlib.gz',
  'libcompiler_builtins-a99947d020d809d6.rlib.gz',
  'libcore-4b8e8a815d049db3.rlib.gz',
  'libgetopts-bbb75529e85d129d.rlib.gz',
  'libgimli-598847d27d7a3cbf.rlib.gz',
  'libhashbrown-d2ff91fdf93cacb2.rlib.gz',
  'liblibc-dc63949c664c3fce.rlib.gz',
  'libmemchr-2d3a423be1a6cb96.rlib.gz',
  'libminiz_oxide-b109506a0ccc4c6a.rlib.gz',
  'libobject-7b48def7544c748b.rlib.gz',
  'libpanic_abort-c93441899b93b849.rlib.gz',
  'libpanic_unwind-11d9ba05b60bf694.rlib.gz',
  'libproc_macro-1a7f7840bb9983dc.rlib.gz',
  'librustc_demangle-59342a335246393d.rlib.gz',
  'librustc_std_workspace_alloc-552b185085090ff6.rlib.gz',
  'librustc_std_workspace_core-5d8a121daa7eeaa9.rlib.gz',
  'librustc_std_workspace_std-97f43841ce452f7d.rlib.gz',
  'libstd-bdedb7706a556da2.rlib.gz',
  'libstd-bdedb7706a556da2.so.gz',
  'libstd_detect-cca21eebc4281add.rlib.gz',
  'libsysroot-f654e185be3ffebd.rlib.gz',
  'libtest-f06fa3fbc201c558.rlib.gz',
  'libunicode_width-19a0dcd589fa0877.rlib.gz',
  'libunwind-747b693f90af9445.rlib.gz',
]

let runtimeId = ''
let miriModule
let sysroot
let wasiShim

function post(message) {
  globalThis.postMessage({ source, runtimeId, ...message })
}

function textOf(value) {
  return value instanceof Error ? value.message : String(value)
}

function emitOutput(cellId, output) {
  post({ type: 'output', cellId, output })
}

function emitError(cellId, error) {
  const text = textOf(error)
  emitOutput(cellId, { type: 'error', ename: 'RustError', evalue: text, traceback: text })
}

function gzipSourceName(filename) {
  return filename.endsWith('.gz') ? filename.slice(0, -3) : filename
}

function assetNeedle(filename) {
  const sourceName = gzipSourceName(filename)
  const extIndex = sourceName.lastIndexOf('.')
  return extIndex === -1 ? sourceName : sourceName.slice(0, extIndex)
}

function assetFor(assets, filename) {
  const needle = assetNeedle(filename)
  const asset = assets.find(item => item.includes(needle))
  if (!asset) throw new Error(`Rust runtime asset is missing ${gzipSourceName(filename)}`)
  return asset
}

async function fetchGzipBytes(url, label) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Rust ${label} request failed with ${response.status}`)
  if (typeof DecompressionStream !== 'function') {
    throw new Error('Rust runtime needs DecompressionStream for compressed sysroot assets')
  }
  if (!response.body) throw new Error(`Rust ${label} response has no body`)
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

function rustProgram(code) {
  const body = code.trim()
  return [
    'fn main() {',
    'let _code = (|| {',
    body,
    '})();',
    'if std::any::Any::type_id(&_code) != std::any::TypeId::of::<()>() { println!("{_code:?}") }',
    '}',
    '',
  ].join('\n')
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

function runtimeWasiShim() {
  if (!wasiShim) throw new Error('Rust WASI shim is not ready')
  return wasiShim
}

function runtimeSysroot() {
  if (!sysroot) throw new Error('Rust sysroot is not ready')
  return sysroot
}

function runtimeMiriModule() {
  if (!miriModule) throw new Error('Rust Miri module is not ready')
  return miriModule
}

function emitCaptured(cellId, name, stdio) {
  const text = stdio.text()
  if (text.length > 0) {
    emitOutput(cellId, { type: 'stream', name, text })
  }
}

async function sysrootFile(url, filename, File) {
  return [
    gzipSourceName(filename),
    new File(await fetchGzipBytes(url, filename), { readonly: true }),
  ]
}

async function buildSysroot(assets, wasiShim) {
  const { Directory, PreopenDirectory } = wasiShim
  const files = await Promise.all(
    sysrootFiles.map(filename =>
      sysrootFile(assetFor(assets, filename), filename, wasiShim.File).finally(() =>
        post({ type: 'status', text: `loaded Rust ${gzipSourceName(filename)}` }),
      ),
    ),
  )
  return new PreopenDirectory('/sysroot', [
    [
      'lib',
      new Directory([
        [
          'rustlib',
          new Directory([
            ['wasm32-wasi', new Directory([['lib', new Directory([])]])],
            ['x86_64-unknown-linux-gnu', new Directory([['lib', new Directory(files)]])],
          ]),
        ],
      ]),
    ],
  ])
}

async function init(message) {
  const assets = Array.isArray(message.assets)
    ? message.assets.filter(item => typeof item === 'string')
    : []
  post({ type: 'status', text: 'loading Rust runtime pack' })
  wasiShim = await import(assetFor(assets, 'browser-wasi-shim.mjs'))
  const [miriBytes, nextSysroot] = await Promise.all([
    fetchGzipBytes(assetFor(assets, 'miri.opt.1718474653.wasm.gz'), 'Miri wasm'),
    buildSysroot(assets, wasiShim),
  ])
  miriModule = await WebAssembly.compile(miriBytes)
  sysroot = nextSysroot
  post({ type: 'ready' })
}

async function run(message) {
  const cellId = typeof message.cellId === 'string' ? message.cellId : ''
  const code = typeof message.code === 'string' ? message.code : ''
  if (!cellId) return
  let failed = false
  const { Fd, File, PreopenDirectory, WASI } = runtimeWasiShim()
  const stdin = captureStdin(Fd)
  const stdout = captureStdio(Fd)
  const stderr = captureStdio(Fd)
  const tmp = new PreopenDirectory('/tmp', [])
  const root = new PreopenDirectory('/', [
    ['main.rs', new File(encoder.encode(rustProgram(code)), { readonly: true })],
  ])
  const wasi = new WASI(
    [
      'miri',
      '--sysroot',
      '/sysroot',
      'main.rs',
      '--target',
      'x86_64-unknown-linux-gnu',
      '-Zmir-opt-level=3',
      '-Zmiri-ignore-leaks',
      '-Zmiri-permissive-provenance',
      '-Zmiri-preemption-rate=0',
      '-Zmiri-disable-alignment-check',
      '-Zmiri-disable-data-race-detector',
      '-Zmiri-disable-stacked-borrows',
      '-Zmiri-disable-validation',
      '-Zmir-emit-retag=false',
      '-Zmiri-disable-isolation',
      '-Zmiri-panic-on-unsupported',
      '--color=never',
    ],
    [],
    [stdin, stdout, stderr, tmp, runtimeSysroot(), root],
    { debug: false },
  )
  try {
    let instance
    let nextThreadId = 1
    instance = await WebAssembly.instantiate(runtimeMiriModule(), {
      env: { memory: new WebAssembly.Memory({ initial: 256, maximum: 4096, shared: false }) },
      wasi: {
        'thread-spawn': startArg => {
          const threadId = nextThreadId
          nextThreadId += 1
          instance.exports.wasi_thread_start(threadId, startArg)
          return threadId
        },
      },
      wasi_snapshot_preview1: wasi.wasiImport,
    })
    const exitCode = wasi.start(instance)
    emitCaptured(cellId, 'stdout', stdout)
    emitCaptured(cellId, 'stderr', stderr)
    const stderrText = stderr.text()
    failed = exitCode !== 0 || stderrText.length > 0
  } catch (error) {
    failed = true
    emitCaptured(cellId, 'stdout', stdout)
    emitCaptured(cellId, 'stderr', stderr)
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
      post({ type: 'status', text: `Rust runtime init failed: ${textOf(error)}` })
      throw error
    })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    void run(message)
  }
})
