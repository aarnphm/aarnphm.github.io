import { type Device, init, defaultDevice, nn, numpy as np } from '@jax-js/jax'
import { PACE_FEATURE_NAMES, type PaceFeatureVector } from './pace-features'
import { type UnknownRecord, isRecord, readNumber, readString } from './type-guards'

type Tensor = ReturnType<typeof np.array>

export interface PaceTensor {
  shape: number[]
  data: Float32Array
}

export interface PaceBackbone {
  kind: string
  tRefS: number
  riegelK: Record<string, number>
  distanceIndex: number
  sportIndices: Record<string, number>
}

export interface PaceManifestOutput {
  muIndex: number
  varIndex: number
  varTransform: string
  varEps: number
  scaleFeature: string
  backbone: PaceBackbone | null
}

export interface PaceManifestArch {
  layers: number
  hidden: number
  activation: string
  layerNormEps: number
  members: number
}

export interface PaceGolden {
  raw: number[]
  presence: number[]
  mu: number
  sigma: number
}

export interface PaceManifest {
  schemaVersion: number
  version: number
  target: string
  featureNames: string[]
  dFeatures: number
  dIn: number
  standardize: { mu: Float32Array; sigma: Float32Array }
  impute: Float32Array
  output: PaceManifestOutput
  arch: PaceManifestArch
  sha256: string
  golden: PaceGolden[]
}

export interface PacePrediction {
  mu: number
  sigma: number
}

interface Member {
  fc1w: Tensor
  fc1b: Tensor
  lnw: Tensor | null
  lnb: Tensor | null
  fc2w: Tensor | null
  fc2b: Tensor | null
}

function softplus(x: number): number {
  return x > 20 ? x : Math.log1p(Math.exp(x))
}

function numberRecord(value: unknown): Record<string, number> {
  const out: Record<string, number> = {}
  if (isRecord(value))
    for (const [k, v] of Object.entries(value)) if (typeof v === 'number') out[k] = v
  return out
}

function parseBackbone(value: unknown): PaceBackbone | null {
  if (!isRecord(value)) return null
  const tRefS = readNumber(value, 'tRefS')
  const distanceIndex = readNumber(value, 'distanceIndex')
  if (tRefS == null || distanceIndex == null) return null
  return {
    kind: readString(value, 'kind') ?? 'riegel',
    tRefS,
    riegelK: numberRecord(value.riegelK),
    distanceIndex,
    sportIndices: numberRecord(value.sportIndices),
  }
}

function riegelKForRaw(raw: Float32Array, bb: PaceBackbone): number {
  for (const [sport, idx] of Object.entries(bb.sportIndices))
    if (raw[idx] > 0.5) return bb.riegelK[sport] ?? 1.06
  return 1.06
}

function floatArray(value: unknown): Float32Array | null {
  if (!Array.isArray(value)) return null
  const out = new Float32Array(value.length)
  for (let i = 0; i < value.length; i++) {
    const n = value[i]
    if (typeof n !== 'number') return null
    out[i] = n
  }
  return out
}

export function parseSafetensors(buf: ArrayBuffer): Map<string, PaceTensor> {
  const view = new DataView(buf)
  const headerLen = Number(view.getBigUint64(0, true))
  const header: unknown = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen)))
  if (!isRecord(header)) throw new Error('pace-model: safetensors header is not an object')
  const base = 8 + headerLen
  const out = new Map<string, PaceTensor>()
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue
    if (!isRecord(meta)) continue
    if (readString(meta, 'dtype') !== 'F32') throw new Error(`pace-model: ${name} is not F32`)
    const shape = Array.isArray(meta.shape) ? meta.shape.map(Number) : []
    const offsets = Array.isArray(meta.data_offsets) ? meta.data_offsets.map(Number) : null
    if (!offsets || offsets.length !== 2)
      throw new Error(`pace-model: ${name} missing data_offsets`)
    const data = new Float32Array(buf.slice(base + offsets[0], base + offsets[1]))
    out.set(name, { shape, data })
  }
  return out
}

export function parseManifest(value: unknown): PaceManifest {
  if (!isRecord(value)) throw new Error('pace-model: manifest is not an object')
  const featureNames = Array.isArray(value.featureNames) ? value.featureNames.map(String) : []
  if (featureNames.length === 0 || featureNames.length > PACE_FEATURE_NAMES.length)
    throw new Error('pace-model: manifest featureNames length out of range')
  for (let i = 0; i < featureNames.length; i++)
    if (featureNames[i] !== PACE_FEATURE_NAMES[i])
      throw new Error(`pace-model: featureNames[${i}] mismatch: ${featureNames[i]}`)

  const std = isRecord(value.standardize) ? value.standardize : {}
  const mu = floatArray(std.mu)
  const sigma = floatArray(std.sigma)
  const impute = floatArray(value.impute)
  const out = isRecord(value.output) ? value.output : {}
  const arch = isRecord(value.arch) ? value.arch : {}
  if (!mu || !sigma || !impute) throw new Error('pace-model: manifest standardize/impute malformed')

  const golden: PaceGolden[] = []
  if (Array.isArray(value.golden))
    for (const g of value.golden) {
      if (!isRecord(g)) continue
      const raw = floatArray(g.raw)
      const presence = floatArray(g.presence)
      const gmu = readNumber(g, 'mu')
      const gsigma = readNumber(g, 'sigma')
      if (raw && presence && gmu != null && gsigma != null)
        golden.push({ raw: [...raw], presence: [...presence], mu: gmu, sigma: gsigma })
    }

  return {
    schemaVersion: readNumber(value, 'schemaVersion') ?? 0,
    version: readNumber(value, 'version') ?? 0,
    target: readString(value, 'target') ?? 'pace',
    featureNames,
    dFeatures: readNumber(value, 'dFeatures') ?? mu.length / 2,
    dIn: readNumber(value, 'dIn') ?? mu.length,
    standardize: { mu, sigma },
    impute,
    output: {
      muIndex: readNumber(out, 'muIndex') ?? 0,
      varIndex: readNumber(out, 'varIndex') ?? 1,
      varTransform: readString(out, 'varTransform') ?? 'softplus',
      varEps: readNumber(out, 'varEps') ?? 1e-6,
      scaleFeature: readString(out, 'scaleFeature') ?? 'vthr',
      backbone: parseBackbone(out.backbone),
    },
    arch: {
      layers: readNumber(arch, 'layers') ?? 0,
      hidden: readNumber(arch, 'hidden') ?? 0,
      activation: readString(arch, 'activation') ?? 'gelu_tanh',
      layerNormEps: readNumber(arch, 'layerNormEps') ?? 1e-5,
      members: readNumber(arch, 'members') ?? 1,
    },
    sha256: readString(value, 'sha256') ?? '',
    golden,
  }
}

let backendReady: Promise<Device> | null = null

export function ensureBackend(preferWebGpu: boolean): Promise<Device> {
  backendReady ??= init('cpu', 'wasm', ...(preferWebGpu ? (['webgpu'] as const) : [])).then(
    devices => {
      const pick: Device =
        preferWebGpu && devices.includes('webgpu')
          ? 'webgpu'
          : devices.includes('wasm')
            ? 'wasm'
            : devices[0]
      defaultDevice(pick)
      return pick
    },
  )
  return backendReady
}

export class PaceModel {
  private constructor(
    readonly manifest: PaceManifest,
    private readonly members: Member[],
    private readonly scaleIndex: number,
    private readonly backbone: PaceBackbone | null,
  ) {}

  static load(manifest: PaceManifest, tensors: Map<string, PaceTensor>): PaceModel {
    const tensor = (name: string): Tensor => {
      const t = tensors.get(name)
      if (!t) throw new Error(`pace-model: missing tensor ${name}`)
      return np.array(t.data as Float32Array<ArrayBuffer>).reshape(t.shape)
    }
    const members: Member[] = []
    for (let m = 0; m < manifest.arch.members; m++) {
      const p = `member.${m}.`
      members.push({
        fc1w: tensor(`${p}fc1.weight`),
        fc1b: tensor(`${p}fc1.bias`),
        lnw: manifest.arch.layers > 0 ? tensor(`${p}ln.weight`) : null,
        lnb: manifest.arch.layers > 0 ? tensor(`${p}ln.bias`) : null,
        fc2w: manifest.arch.layers > 0 ? tensor(`${p}fc2.weight`) : null,
        fc2b: manifest.arch.layers > 0 ? tensor(`${p}fc2.bias`) : null,
      })
    }
    const scaleIndex = PACE_FEATURE_NAMES.indexOf(
      manifest.output.scaleFeature as (typeof PACE_FEATURE_NAMES)[number],
    )
    return new PaceModel(
      manifest,
      members,
      scaleIndex < 0 ? 17 : scaleIndex,
      manifest.output.backbone,
    )
  }

  private standardize(raw: Float32Array, presence: Float32Array): Tensor {
    const { mu, sigma } = this.manifest.standardize
    const df = this.manifest.dFeatures
    const xin = new Float32Array(this.manifest.dIn)
    for (let i = 0; i < df; i++) {
      const filled = presence[i] > 0 ? raw[i] : this.manifest.impute[i]
      xin[i] = (filled - mu[i]) / sigma[i]
      xin[df + i] = (presence[i] - mu[df + i]) / sigma[df + i]
    }
    const d = this.backbone?.distanceIndex
    if (d != null) {
      xin[d] = 0
      xin[df + d] = 0
    }
    return np.array(xin)
  }

  private memberOutput(m: Member, x: Tensor): Tensor {
    if (m.lnw === null || m.lnb === null || m.fc2w === null || m.fc2b === null)
      return np.add(np.matvec(m.fc1w.ref, x), m.fc1b.ref)
    let h = np.add(np.matvec(m.fc1w.ref, x), m.fc1b.ref)
    h = nn.standardize(h, -1, { epsilon: this.manifest.arch.layerNormEps })
    h = np.add(np.multiply(h, m.lnw.ref), m.lnb.ref)
    h = nn.gelu(h)
    return np.add(np.matvec(m.fc2w.ref, h), m.fc2b.ref)
  }

  async forwardIntensity(raw: Float32Array, presence: Float32Array): Promise<PacePrediction> {
    const x = this.standardize(raw, presence)
    const { muIndex, varIndex, varEps } = this.manifest.output
    let sumMu = 0
    let sumMuSq = 0
    let sumVar = 0
    for (const m of this.members) {
      const out = this.memberOutput(m, x.ref)
      const vals = await out.ref.data()
      out.dispose()
      const mu = vals[muIndex]
      const variance = softplus(vals[varIndex]) + varEps
      sumMu += mu
      sumMuSq += mu * mu
      sumVar += variance
    }
    x.dispose()
    const n = this.members.length
    const muBar = sumMu / n
    const varBar = sumVar / n + (sumMuSq / n - muBar * muBar)
    return { mu: muBar, sigma: Math.sqrt(Math.max(varBar, 0)) }
  }

  async predict(raw: Float32Array, presence: Float32Array): Promise<PacePrediction> {
    const intensity = await this.forwardIntensity(raw, presence)
    const bb = this.backbone
    if (bb) {
      const vThr = raw[this.scaleIndex]
      const dist = raw[bb.distanceIndex]
      const dRef = (vThr * bb.tRefS) / 1000
      const vBackbone =
        dRef > 0 && dist > 0 ? vThr * Math.pow(dist / dRef, 1 - riegelKForRaw(raw, bb)) : vThr
      const v = vBackbone * Math.exp(intensity.mu)
      return { mu: v, sigma: v * intensity.sigma }
    }
    const scale = raw[this.scaleIndex]
    return { mu: intensity.mu * scale, sigma: intensity.sigma * scale }
  }

  predictVector(fv: PaceFeatureVector): Promise<PacePrediction> {
    return this.predict(fv.raw, fv.presence)
  }

  async checkGolden(tol: number): Promise<{ ok: boolean; maxErr: number }> {
    let maxErr = 0
    for (const g of this.manifest.golden) {
      const pred = await this.forwardIntensity(
        Float32Array.from(g.raw),
        Float32Array.from(g.presence),
      )
      maxErr = Math.max(maxErr, Math.abs(pred.mu - g.mu), Math.abs(pred.sigma - g.sigma))
    }
    return { ok: maxErr <= tol, maxErr }
  }
}

export function isPaceManifestJson(value: unknown): value is UnknownRecord {
  return isRecord(value) && Array.isArray(value.featureNames)
}
