import type { CodeWorkerRequest, CodeWorkerResponse } from '../util/code-search-protocol'
import { CodeSearchEngine } from '../util/code-search-engine'

let engine: CodeSearchEngine | undefined
let manifestUrl: string | undefined
let active: AbortController | undefined
const send = (message: CodeWorkerResponse) => self.postMessage(message)

self.addEventListener('message', async (event: MessageEvent<CodeWorkerRequest>) => {
  const message = event.data
  if (message.type !== 'search') return
  active?.abort()
  const controller = new AbortController()
  active = controller
  if (!engine || manifestUrl !== message.manifestUrl) {
    engine = new CodeSearchEngine(message.manifestUrl)
    manifestUrl = message.manifestUrl
  }
  try {
    const result = await engine.search(message.query, controller.signal, progress => {
      if (!controller.signal.aborted) send({ type: 'progress', id: message.id, progress })
    })
    if (!controller.signal.aborted) send({ type: 'result', id: message.id, result })
  } catch (error) {
    if (!controller.signal.aborted)
      send({
        type: 'error',
        id: message.id,
        message: error instanceof Error ? error.message : String(error),
        name: error instanceof Error ? error.name : 'Error',
      })
  }
})
