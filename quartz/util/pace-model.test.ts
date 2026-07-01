import assert from 'node:assert/strict'
import { test } from 'node:test'
import { PACE_FEATURE_NAMES } from './pace-features'
import {
  type PaceTensor,
  PaceModel,
  ensureBackend,
  parseManifest,
  parseSafetensors,
} from './pace-model'

function buildSafetensors(
  tensors: { name: string; shape: number[]; data: Float32Array }[],
): ArrayBuffer {
  const header: Record<string, unknown> = {}
  let offset = 0
  for (const t of tensors) {
    const bytes = t.data.byteLength
    header[t.name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + bytes] }
    offset += bytes
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header))
  const buf = new ArrayBuffer(8 + headerBytes.length + offset)
  new DataView(buf).setBigUint64(0, BigInt(headerBytes.length), true)
  new Uint8Array(buf, 8, headerBytes.length).set(headerBytes)
  let p = 8 + headerBytes.length
  for (const t of tensors) {
    new Uint8Array(buf, p, t.data.byteLength).set(
      new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength),
    )
    p += t.data.byteLength
  }
  return buf
}

function syntheticManifest(): Record<string, unknown> {
  return {
    schemaVersion: 1,
    version: 1,
    target: 'pace',
    featureNames: [...PACE_FEATURE_NAMES],
    dFeatures: 19,
    dIn: 38,
    standardize: { mu: Array<number>(38).fill(0), sigma: Array<number>(38).fill(1) },
    impute: Array<number>(19).fill(0),
    output: {
      muIndex: 0,
      varIndex: 1,
      varTransform: 'softplus',
      varEps: 1e-6,
      scaleFeature: 'vthr',
    },
    arch: { layers: 0, hidden: 0, activation: 'gelu_tanh', layerNormEps: 1e-5, members: 1 },
    sha256: '',
    golden: [],
  }
}

test('parseManifest rejects featureNames mismatch', () => {
  const bad = syntheticManifest()
  ;(bad.featureNames as string[])[5] = 'wrong'
  assert.throws(() => parseManifest(bad), /featureNames\[5\] mismatch/)
})

test('parseManifest rejects wrong featureNames length', () => {
  const bad = syntheticManifest()
  bad.featureNames = ['only', 'three', 'names']
  assert.throws(() => parseManifest(bad), /length mismatch/)
})

test('parseSafetensors round-trips an F32 tensor', () => {
  const data = new Float32Array([1, 2, 3, 4, 5, 6])
  const buf = buildSafetensors([{ name: 'member.0.fc1.weight', shape: [2, 3], data }])
  const parsed: Map<string, PaceTensor> = parseSafetensors(buf)
  const t = parsed.get('member.0.fc1.weight')
  assert.ok(t)
  assert.deepEqual(t.shape, [2, 3])
  assert.deepEqual(Array.from(t.data), [1, 2, 3, 4, 5, 6])
})

test('parseSafetensors rejects non-F32 dtype', () => {
  const header = { x: { dtype: 'F16', shape: [1], data_offsets: [0, 2] } }
  const hb = new TextEncoder().encode(JSON.stringify(header))
  const buf = new ArrayBuffer(8 + hb.length + 2)
  new DataView(buf).setBigUint64(0, BigInt(hb.length), true)
  new Uint8Array(buf, 8, hb.length).set(hb)
  assert.throws(() => parseSafetensors(buf), /not F32/)
})

test('forwardIntensity reproduces a hand-computed linear member', async () => {
  await ensureBackend(false)
  const w = new Float32Array(2 * 38)
  w[0] = 1
  const b = new Float32Array([0.5, 0])
  const buf = buildSafetensors([
    { name: 'member.0.fc1.weight', shape: [2, 38], data: w },
    { name: 'member.0.fc1.bias', shape: [2], data: b },
  ])
  const model = PaceModel.load(parseManifest(syntheticManifest()), parseSafetensors(buf))
  const raw = new Float32Array(19)
  raw[0] = 2
  const presence = new Float32Array(19).fill(1)
  const { mu, sigma } = await model.forwardIntensity(raw, presence)
  assert.ok(Math.abs(mu - 2.5) < 1e-5, `mu=${mu}`)
  assert.ok(Math.abs(sigma - Math.sqrt(Math.log1p(1) + 1e-6)) < 1e-5, `sigma=${sigma}`)
})
