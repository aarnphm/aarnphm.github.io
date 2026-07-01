import { PaceModel, ensureBackend, parseManifest, parseSafetensors } from '../util/pace-model'
import { isRecord, readString } from '../util/type-guards'

export {}

type LoadMessage = { type: 'load'; base: string; family: string }
type PredictMessage = { type: 'predict'; id: number; raw: number[]; presence: number[] }
type WorkerMessage = LoadMessage | PredictMessage

const GOLDEN_TOL_WASM = 1e-4
const GOLDEN_TOL_WEBGPU = 5e-2

let model: PaceModel | null = null
let device = 'wasm'

async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`)
  return res.json()
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buf)
  return Array.from(new Uint8Array(digest))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('')
}

async function handleLoad(base: string, family: string): Promise<void> {
  device = await ensureBackend(true)
  const latest = await fetchJson(`${base}/models/${family}/latest.json`)
  if (!isRecord(latest)) throw new Error('latest.json malformed')
  const manifestKey = readString(latest, 'manifest')
  const weightsKey = readString(latest, 'weights')
  if (!manifestKey || !weightsKey) throw new Error('latest.json missing keys')

  const manifest = parseManifest(await fetchJson(`${base}/${manifestKey}`))
  const weightsRes = await fetch(`${base}/${weightsKey}`)
  if (!weightsRes.ok) throw new Error(`weights ${weightsRes.status}`)
  const buf = await weightsRes.arrayBuffer()
  if (manifest.sha256 && (await sha256Hex(buf)) !== manifest.sha256)
    throw new Error('safetensors sha256 mismatch')

  const candidate = PaceModel.load(manifest, parseSafetensors(buf))
  const tol = device === 'webgpu' ? GOLDEN_TOL_WEBGPU : GOLDEN_TOL_WASM
  const gate = await candidate.checkGolden(tol)
  if (!gate.ok) throw new Error(`golden parity ${gate.maxErr} > ${tol}`)

  model = candidate
  self.postMessage({ type: 'loaded', ok: true, version: manifest.version, device })
}

async function handlePredict(msg: PredictMessage): Promise<void> {
  if (!model) {
    self.postMessage({ type: 'prediction', id: msg.id, ok: false })
    return
  }
  const { mu, sigma } = await model.predict(
    Float32Array.from(msg.raw),
    Float32Array.from(msg.presence),
  )
  const ok = Number.isFinite(mu) && Number.isFinite(sigma)
  self.postMessage({ type: 'prediction', id: msg.id, ok, mu, sigma })
}

self.onmessage = (event: MessageEvent<WorkerMessage>): void => {
  const msg = event.data
  const run =
    msg.type === 'load'
      ? handleLoad(msg.base, msg.family)
      : msg.type === 'predict'
        ? handlePredict(msg)
        : Promise.resolve()
  run.catch((err: unknown) => {
    const reason = err instanceof Error ? err.message : String(err)
    if (msg.type === 'load') self.postMessage({ type: 'loaded', ok: false, reason })
    else if (msg.type === 'predict') self.postMessage({ type: 'prediction', id: msg.id, ok: false })
  })
}
