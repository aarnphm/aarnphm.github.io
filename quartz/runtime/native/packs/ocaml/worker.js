const source = 'quartz-notebook-runtime'
let runtimeId = ''
let evaluator
let currentCellId = ''
let stderrText = ''

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
  emitOutput(cellId, { type: 'error', ename: 'OCamlError', evalue: text, traceback: text })
}

globalThis.quartzNativeOcamlStream = (name, text) => {
  if (!currentCellId || text.length === 0) return
  if (name === 'stderr') stderrText += text
  emitOutput(currentCellId, { type: 'stream', name, text })
}

async function loadScript(url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`OCaml runtime script request failed with ${response.status}`)
  const code = await response.text()
  globalThis.Function(code)()
}

async function init(message) {
  const assets = Array.isArray(message.assets)
    ? message.assets.filter(item => typeof item === 'string')
    : []
  const scriptUrl = assets.find(item => item.includes('/toplevel'))
  if (!scriptUrl) throw new Error('OCaml runtime asset is missing toplevel.js')
  await loadScript(scriptUrl)
  evaluator = globalThis.quartzNativeOcaml
  if (!evaluator || typeof evaluator.execute !== 'function') {
    throw new Error('OCaml runtime did not expose quartzNativeOcaml.execute')
  }
  post({ type: 'ready' })
}

function run(message) {
  const cellId = typeof message.cellId === 'string' ? message.cellId : ''
  const code = typeof message.code === 'string' ? message.code : ''
  if (!cellId) return
  currentCellId = cellId
  stderrText = ''
  let failed = false
  try {
    if (!evaluator || typeof evaluator.execute !== 'function') {
      throw new Error('OCaml runtime is not ready')
    }
    const text = String(evaluator.execute(code))
    if (text.length > 0) emitOutput(cellId, { type: 'text', text })
    if (stderrText.length > 0) failed = true
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
      post({ type: 'status', text: `OCaml runtime init failed: ${textOf(error)}` })
      throw error
    })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    run(message)
  }
})
