import type { LinkItem } from '../../../util/latent/markup'
import type { RailLayout } from '../../../util/latent/rail'
import type { LatentWorkerRequest, LatentWorkerResponse } from '../../../workers/latent.worker'
import type { ToastHandle } from '../toast'
import {
  blockName,
  blockOf,
  decodeKnnBlock,
  partName,
  type LatentChunk,
  type LatentDoc,
  type LatentKind,
  type LatentPreview,
  type LatentRowLinks,
  type LatentText,
  type LatentTextPart,
  type LatentTrackDoc,
} from '../../../util/latent/types'

export interface Runtime {
  root: HTMLElement
  base: string
  k: number
  block: number
  axes: number
  kinds: LatentKind[]
  scope: string[]
  signal: AbortSignal
  docs: LatentDoc[]
  track: LatentTrackDoc[]
  knn: Map<number, Promise<LatentRowLinks[]>>
  prev: Map<number, Promise<LatentPreview[]>>
  text: Map<number, Promise<LatentText>>
  parts: Map<string, Promise<LatentChunk[]>>
  mounting: Map<number, Promise<HTMLElement | null>>
  regions: Map<string, Promise<Record<string, { bbox: number[]; text: string }[]> | null>>
  docByRow: LatentDoc[]
  focus: number
  open: number
  rail: HTMLElement
  reader: HTMLElement
  /** the notes' KaTeX macros, from the build */
  macros: Record<string, string>
  links: HTMLElement
  probe: HTMLElement
  bench: Bench | null
  /** the vectors worker, shared by the bench and the query */
  vec: { worker: Worker; seq: number } | null
  semantic: Worker | null
  model: string
  /** a typed query pins the links pane to its hits until cleared */
  query: { text: string; items: LinkItem[] } | null
  railLayout: RailLayout | null
  /** rail viewBox height: the pane's aspect ratio in rail units, so the track spans the pane */
  railHeight: number
  railItems: LinkItem[]
  /** the bottom-right toast that carries the search phases; one per page, text changes in place */
  toast: ToastHandle | null
  /** the text the latest query was started for; older queries drop their results */
  wanted: string
}

export interface Bench {
  worker: Worker
  seq: number
  level: 'docs' | 'rows'
  coords: Float32Array | null
  extent: number
  yaw: number
  pitch: number
  zoom: number
  // screen offset of the origin from the canvas centre, css px
  panX: number
  panY: number
  kinds: Uint8Array | null
  axes: number[]
  meta: {
    scale: number
    mean: number[]
    axes: { id: string; name: string; dir: number[] }[]
    rows: number
    docs: number
  } | null
  frame: number
  hover: number
  // canvas size in css px; the backing store is w·dpr × h·dpr so points and labels stay sharp
  w: number
  h: number
  dpr: number
}

export async function fetchJson<T>(rt: Pick<Runtime, 'signal'>, url: string): Promise<T> {
  const response = await fetch(url, { signal: rt.signal })
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  return (await response.json()) as T
}

export async function fetchBytes(rt: Pick<Runtime, 'signal'>, url: string): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal: rt.signal })
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  return response.arrayBuffer()
}

export function knnOf(rt: Runtime, row: number): Promise<LatentRowLinks> {
  const b = blockOf(row, rt.block)
  let block = rt.knn.get(b)
  if (!block) {
    block = fetchBytes(rt, `${rt.base}/knn/${blockName(b)}.bin`).then(bytes =>
      decodeKnnBlock(bytes, b, { k: rt.k, axes: rt.axes, block: rt.block }),
    )
    rt.knn.set(b, block)
  }
  return block.then(rows => rows[row - b * rt.block])
}

export function previewOf(rt: Runtime, row: number): Promise<LatentPreview> {
  const b = blockOf(row, rt.block)
  let block = rt.prev.get(b)
  if (!block) {
    block = fetchJson<LatentPreview[]>(rt, `${rt.base}/prev/${blockName(b)}.json`)
    rt.prev.set(b, block)
  }
  return block.then(rows => rows[row - b * rt.block])
}

/** The head file: document meta plus part 0. */
export function textOf(rt: Runtime, doc: number): Promise<LatentText> {
  let text = rt.text.get(doc)
  if (!text) {
    text = fetchJson<LatentText>(rt, `${rt.base}/text/${partName(doc, 0)}`)
    rt.text.set(doc, text)
  }
  return text
}

/** Chunks of one part of a document. */
export function partOf(rt: Runtime, doc: number, part: number): Promise<LatentChunk[]> {
  if (part === 0) return textOf(rt, doc).then(t => t.chunks)
  const key = `${doc}:${part}`
  let chunks = rt.parts.get(key)
  if (!chunks) {
    chunks = fetchJson<LatentTextPart>(rt, `${rt.base}/text/${partName(doc, part)}`).then(
      p => p.chunks,
    )
    rt.parts.set(key, chunks)
  }
  return chunks
}

export function docOfRow(rt: Runtime, row: number): LatentDoc {
  return rt.docByRow[row]
}

// ---- typed query ----------------------------------------------------------------------------

export function vecWorker(rt: Runtime): { worker: Worker; seq: number } {
  if (!rt.vec) {
    const worker = new Worker(new URL('latent.worker.js', import.meta.url), { type: 'module' })
    rt.vec = { worker, seq: 0 }
    rt.signal.addEventListener('abort', () => worker.terminate())
  }
  return rt.vec
}

type VecMessage = LatentWorkerRequest extends infer R
  ? R extends LatentWorkerRequest
    ? Omit<R, 'seq'>
    : never
  : never

export function requestVec(
  rt: Runtime,
  message: VecMessage,
  transfer: Transferable[] = [],
): Promise<LatentWorkerResponse> {
  const vec = vecWorker(rt)
  const seq = ++vec.seq
  return new Promise((resolve, reject) => {
    const onMessage = (event: MessageEvent<LatentWorkerResponse>) => {
      if (event.data.seq !== seq) return
      vec.worker.removeEventListener('message', onMessage)
      if (event.data.type === 'error') reject(new Error(event.data.message))
      else resolve(event.data)
    }
    vec.worker.addEventListener('message', onMessage)
    vec.worker.postMessage({ ...message, seq } as LatentWorkerRequest, transfer)
  })
}

let metaCache: WeakMap<Runtime, Promise<NonNullable<Bench['meta']>>> = new WeakMap()
export function metaOf(rt: Runtime): Promise<NonNullable<Bench['meta']>> {
  let meta = metaCache.get(rt)
  if (!meta) {
    meta = fetchJson<NonNullable<Bench['meta']>>(rt, `${rt.base}/vec/meta.json`)
    metaCache.set(rt, meta)
  }
  return meta
}
