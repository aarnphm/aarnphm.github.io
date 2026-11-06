import { readFile } from 'node:fs/promises'
import path from 'node:path'
import type {
  FeatureDict,
  FeatureState,
} from '../../../components/scripts/latent-client/features-types'
import type {
  Bench,
  LatentSource,
  Runtime,
} from '../../../components/scripts/latent-client/runtime'
import type { Sidecar } from '../sidecar'
import { VectorEngine } from '../vectors'

/**
 * The latent client's runtime, at build time: files come from the sidecar folder instead of
 * `fetch`, and the vector math runs in-process instead of the worker. The features panel
 * builders (latent-client/features-*.ts) take it as they take the browser's, so an embed's
 * column and steered links are what /embeddings renders for the same view. One per sidecar per
 * process: a note with three embeds, or a site with many, opens the 16 MB rows matrix once.
 */
export interface NodeLatent {
  sidecar: Sidecar
  source: LatentSource
  engine: VectorEngine
  /** shared across embeds of this sidecar: feature dictionaries and the tables beside them */
  shared: Pick<FeatureState, 'dicts' | 'channels'> & {
    tables: FeatureState['tables']
    tablesNow: FeatureState['tablesNow']
    tags: FeatureState['tags']
    words: FeatureState['words']
  }
  meta: Promise<NonNullable<Bench['meta']>>
  docByRow: Runtime['docByRow']
}

const cache = new Map<string, NodeLatent>()

export function nodeLatent(sidecar: Sidecar): NodeLatent {
  const key = sidecar.dir
  const known = cache.get(key)
  if (known && known.sidecar === sidecar) return known
  const files = new Map<string, Promise<Buffer>>()
  const file = (url: string): Promise<Buffer> => {
    if (!url.startsWith(`${sidecar.base}/`)) throw new Error(`not in the sidecar: ${url}`)
    const rel = url.slice(sidecar.base.length + 1)
    if (rel.split('/').includes('..')) throw new Error(`not in the sidecar: ${url}`)
    let pending = files.get(rel)
    if (!pending) {
      pending = readFile(path.join(sidecar.dir, rel))
      pending.catch(() => files.delete(rel))
      files.set(rel, pending)
    }
    return pending
  }
  const bytes = async (url: string): Promise<ArrayBuffer> => {
    const buffer = await file(url)
    const copy = new ArrayBuffer(buffer.byteLength)
    new Uint8Array(copy).set(buffer)
    return copy
  }
  const source: LatentSource = {
    json: async <T>(url: string) => JSON.parse((await file(url)).toString('utf8')) as T,
    bytes,
  }
  const docByRow: Runtime['docByRow'] = Array.from({ length: sidecar.index.rows })
  for (const doc of sidecar.docs) docByRow.fill(doc, doc.r0, doc.r0 + doc.n)
  const latent: NodeLatent = {
    sidecar,
    source,
    engine: new VectorEngine(bytes),
    shared: {
      dicts: new Map<string, Promise<FeatureDict>>(),
      channels: new Set(),
      tables: null,
      tablesNow: null,
      tags: null,
      words: null,
    },
    meta: source.json<NonNullable<Bench['meta']>>(`${sidecar.base}/vec/meta.json`),
    docByRow,
  }
  cache.set(key, latent)
  return latent
}

/** A runtime for one embed: the shared caches, its own focus. */
export function nodeRuntime(latent: NodeLatent, focus: number, scope: string[]): Runtime {
  const { sidecar } = latent
  const doc = latent.docByRow[focus]
  const rt = {
    root: null,
    base: sidecar.base,
    k: sidecar.index.k,
    block: sidecar.index.block,
    axes: sidecar.index.axes,
    kinds: sidecar.index.kinds,
    scope,
    signal: new AbortController().signal,
    docs: sidecar.docs,
    track: sidecar.track,
    knn: new Map(),
    prev: new Map(),
    text: new Map(),
    parts: new Map(),
    mounting: new Map(),
    regions: new Map(),
    docByRow: latent.docByRow,
    focus,
    open: doc?.d ?? 0,
    rail: null,
    reader: null,
    macros: {},
    links: null,
    probe: null,
    bench: null,
    vec: null,
    semantic: null,
    model: sidecar.index.model,
    query: null,
    railLayout: null,
    railHeight: 0,
    railItems: [],
    toast: null,
    wanted: '',
    filter: null,
    source: latent.source,
    engine: latent.engine,
    mode: 'embed',
  }
  return rt as unknown as Runtime
}

/** The features state of one embed over the shared caches; `pin` records the steered rows. */
export function nodeFeatureState(
  latent: NodeLatent,
  names: string[],
  mask?: Uint8Array,
): FeatureState & { pinnedRows: [number, number][] | null; pinnedLabel: string } {
  const shared = latent.shared
  const st = {
    refresh() {},
    async pin(label: string, rows: [number, number][]) {
      st.pinnedRows = rows
      st.pinnedLabel = label
    },
    unpin() {},
    mask: async () => mask,
    embed: true,
    inert: true,
    names,
    dicts: shared.dicts,
    // tables/tags/words promise caches go through the shared record so every embed reuses them
    get tags() {
      return shared.tags
    },
    set tags(value) {
      shared.tags = value
    },
    get tables() {
      return shared.tables
    },
    set tables(value) {
      shared.tables = value
    },
    get tablesNow() {
      return shared.tablesNow
    },
    set tablesNow(value) {
      shared.tablesNow = value
    },
    get words() {
      return shared.words
    },
    set words(value) {
      shared.words = value
    },
    selected: null,
    query: null,
    needle: '',
    steer: new Map(),
    steerSeq: 0,
    channels: shared.channels,
    explain: true,
    whyReady: true,
    renderSeq: 0,
    whyCache: new Map(),
    whyKey: '',
    pinned: null,
    dragFrame: null,
    panel: null,
    pinnedRows: null as [number, number][] | null,
    pinnedLabel: '',
  }
  return st as unknown as FeatureState & {
    pinnedRows: [number, number][] | null
    pinnedLabel: string
  }
}
