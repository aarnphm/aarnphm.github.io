import type {
  LexicalWorkerErrorMessage,
  LexicalWorkerReadyMessage,
  LexicalWorkerRequest,
  LexicalWorkerSearchResultMessage,
} from '../components/scripts/lexical-search-types'
import {
  createLexicalDocumentIndex,
  searchLexicalDocumentIndex,
} from '../components/scripts/lexical-search-index'

let index = createLexicalDocumentIndex()
let activeInitSeq = 0
let indexedCount = 0
let ready = false

function postError(message: string, seq?: number) {
  const payload: LexicalWorkerErrorMessage = { type: 'error', seq, message }
  self.postMessage(payload)
}

self.onmessage = (event: MessageEvent<LexicalWorkerRequest>) => {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== self.location.origin) return

  const msg = event.data

  if (msg.type === 'init') {
    try {
      activeInitSeq = msg.seq
      indexedCount = 0
      ready = false
      index = createLexicalDocumentIndex()
    } catch (err) {
      postError(err instanceof Error ? err.message : String(err), msg.seq)
    }
    return
  }

  if (msg.type === 'add') {
    if (msg.seq !== activeInitSeq) return
    try {
      index.add(msg.item)
      indexedCount += 1
    } catch (err) {
      ready = false
      postError(err instanceof Error ? err.message : String(err), msg.seq)
    }
    return
  }

  if (msg.type === 'commit') {
    if (msg.seq !== activeInitSeq) return
    if (indexedCount !== msg.count) {
      ready = false
      postError(
        `lexical worker count mismatch: expected ${msg.count}, indexed ${indexedCount}`,
        msg.seq,
      )
      return
    }
    ready = true
    const response: LexicalWorkerReadyMessage = { type: 'ready', seq: msg.seq, count: indexedCount }
    self.postMessage(response)
    return
  }

  if (msg.type === 'search') {
    if (!ready) {
      postError('lexical worker is not ready for search', msg.seq)
      return
    }
    void searchLexicalDocumentIndex(index, msg.spec)
      .then(results => {
        const response: LexicalWorkerSearchResultMessage = {
          type: 'search-result',
          seq: msg.seq,
          results,
        }
        self.postMessage(response)
      })
      .catch((err: unknown) => {
        postError(err instanceof Error ? err.message : String(err), msg.seq)
      })
  }
}
