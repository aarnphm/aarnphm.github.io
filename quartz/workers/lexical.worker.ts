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
import {
  decodeCompactLexicalIndex,
  importLexicalQuery,
  lexicalPrebuiltKind,
  type DecodedCompactLexicalIndex,
} from '../components/scripts/lexical-search-prebuilt'

let compact: DecodedCompactLexicalIndex | null = null
let index = createLexicalDocumentIndex()
let activeInitSeq = 0
let indexedCount = 0
let ready = false

async function restorePrebuilt(fingerprint: string, count: number, seq: number): Promise<void> {
  let restored = false
  try {
    const manifestUrl = new URL('../searchIndex.json', self.location.href)
    const response = await fetch(manifestUrl, { signal: AbortSignal.timeout(1500) })
    if (!response.ok) throw new Error('Lexical manifest unavailable')
    const manifest = await response.json()
    const resource = manifest?.lexical
    if (
      resource?.kind !== lexicalPrebuiltKind ||
      resource.fingerprint !== fingerprint ||
      typeof resource.path !== 'string' ||
      typeof resource.sha256 !== 'string'
    )
      throw new Error('Lexical snapshot does not match')
    const url = new URL(resource.path, manifestUrl)
    if (url.origin !== self.location.origin) throw new Error('Invalid lexical resource origin')
    const binary = await fetch(url, { signal: AbortSignal.timeout(1500) })
    if (!binary.ok) throw new Error('Lexical snapshot unavailable')
    if (!binary.body) throw new Error('Empty lexical snapshot')
    const bytes = new Uint8Array(
      await new Response(binary.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(),
    )
    const digest = await crypto.subtle.digest('SHA-256', bytes)
    const hash = Array.from(new Uint8Array(digest), byte =>
      byte.toString(16).padStart(2, '0'),
    ).join('')
    if (hash !== resource.sha256) throw new Error('Lexical snapshot checksum mismatch')
    const decoded = decodeCompactLexicalIndex(bytes)
    if (decoded.reg.length !== count) throw new Error('Lexical snapshot count mismatch')
    if (seq !== activeInitSeq) return
    Reflect.apply(index.import, index, ['1.reg', decoded.reg])
    compact = decoded
    indexedCount = count
    ready = true
    restored = true
  } catch {
    // Missing or stale optional artifacts use the original streamed index.
    if (seq === activeInitSeq) {
      compact = null
      index = createLexicalDocumentIndex()
      indexedCount = 0
    }
  }
  if (seq === activeInitSeq) self.postMessage({ type: 'prebuilt', seq, restored })
}

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
      compact = null
      index = createLexicalDocumentIndex()
      if (msg.fingerprint && typeof msg.count === 'number')
        void restorePrebuilt(msg.fingerprint, msg.count, msg.seq)
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
    if (compact) importLexicalQuery(index, compact, msg.spec)
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
