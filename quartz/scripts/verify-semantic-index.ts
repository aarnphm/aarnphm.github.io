import { AutoConfig, AutoModel, AutoTokenizer } from '@huggingface/transformers'
// Checks that the stored document vectors and the browser query encoder live in the same space.
// It re-embeds whole-document rows with the ONNX model the browser loads and compares them with the
// stored rows. A wrong prefix, a model mismatch or shifted rows lower the cosine.
import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { SEMANTIC_ONNX_MODELS, type SemanticQueryDType } from '../util/semantic-search'

type Shard = { path: string; rows: number; rowOffset: number; byteLength: number; sha256?: string }
type Manifest = {
  model: string
  dims: number
  rows: number
  vectors: { dtype: string; shards: Shard[] }
  ids: string[]
  chunkMetadata?: Record<string, unknown>
}

const { values } = parseArgs({
  options: {
    dir: { type: 'string', default: 'public/embeddings' },
    jsonl: { type: 'string', default: 'public/embeddings-text.jsonl' },
    samples: { type: 'string', default: '24' },
    dtype: { type: 'string', default: 'q4' },
    'min-median': { type: 'string', default: '0.95' },
    report: { type: 'string' },
  },
})
const dir = values.dir!
const dtype = values.dtype as SemanticQueryDType
const minMedian = Number(values['min-median'])

const manifest = JSON.parse(await readFile(path.join(dir, 'manifest.json'), 'utf8')) as Manifest
if (!manifest.model.toLowerCase().includes('embeddinggemma')) {
  throw new Error(`document prompts are only implemented for embeddinggemma, got ${manifest.model}`)
}
if (manifest.vectors.dtype !== 'fp32')
  throw new Error(`expected fp32 vectors, got ${manifest.vectors.dtype}`)
const onnxRepo = SEMANTIC_ONNX_MODELS[manifest.model]
if (!onnxRepo) throw new Error(`no ONNX repo mapped for ${manifest.model}`)

const { dims, rows } = manifest
const vectors = new Float32Array(rows * dims)
for (const shard of manifest.vectors.shards) {
  const payload = await readFile(path.join(dir, path.basename(shard.path)))
  if (payload.byteLength !== shard.rows * dims * 4) throw new Error(`${shard.path}: wrong length`)
  if (shard.sha256 && createHash('sha256').update(payload).digest('hex') !== shard.sha256) {
    throw new Error(`${shard.path}: sha256 differs from the manifest`)
  }
  vectors.set(
    new Float32Array(payload.buffer, payload.byteOffset, shard.rows * dims),
    shard.rowOffset * dims,
  )
}
let minNorm = Infinity
let maxNorm = 0
for (let r = 0; r < rows; r++) {
  let sum = 0
  for (let d = 0; d < dims; d++) sum += vectors[r * dims + d] ** 2
  if (!Number.isFinite(sum)) throw new Error(`row ${r} holds NaN or Infinity`)
  minNorm = Math.min(minNorm, Math.sqrt(sum))
  maxNorm = Math.max(maxNorm, Math.sqrt(sum))
}

const documents = new Map<string, { title?: string; text: string }>()
for (const line of (await readFile(values.jsonl!, 'utf8')).split('\n')) {
  if (line.trim()) documents.set(JSON.parse(line).slug, JSON.parse(line))
}
// Chunk rows need the chunker to rebuild their text. Whole-document rows are exactly the JSONL text.
const candidates = manifest.ids
  .map((id, row) => ({ id, row }))
  .filter(({ id }) => !manifest.chunkMetadata?.[id] && documents.has(id))
const count = Math.min(Number(values.samples), candidates.length)
if (count === 0)
  throw new Error('no whole-document rows match the JSONL; rebuild it from the same content')
const sampled = Array.from(
  { length: count },
  (_, i) => candidates[Math.floor((i * candidates.length) / count)],
)

const config = await AutoConfig.from_pretrained(onnxRepo)
if (config.model_type === 'embedding_gemma2') {
  Object.assign(config, { vision_config: null, audio_config: null })
}
const tokenizer = await AutoTokenizer.from_pretrained(onnxRepo)
const model = await AutoModel.from_pretrained(onnxRepo, { config, dtype })

async function embed(text: string): Promise<Float32Array> {
  const outputs = await model(tokenizer([text], { padding: true }))
  const vec = Float32Array.from(outputs.sentence_embedding.data.slice(0, dims) as Float32Array)
  const norm = Math.hypot(...vec)
  return vec.map(v => v / norm)
}

const dot = (a: Float32Array, row: number) => {
  let sum = 0
  for (let d = 0; d < dims; d++) sum += a[d] * vectors[row * dims + d]
  return sum
}
const quantile = (xs: number[], q: number) =>
  [...xs].sort((a, b) => a - b)[Math.floor(q * (xs.length - 1))]

const prompts: Record<string, (title: string, text: string) => string> = {
  document: (title, text) => `title: ${title} | text: ${text}`,
  bare: (_, text) => text,
  query: (_, text) => `task: search result | query: ${text}`,
  doubled: (title, text) => `title: none | text: title: ${title} | text: ${text}`,
}
const cosines: Record<string, number[]> = Object.fromEntries(Object.keys(prompts).map(k => [k, []]))
let top1 = 0
let top5 = 0
for (const { id, row } of sampled) {
  const { title, text } = documents.get(id)!
  for (const [name, build] of Object.entries(prompts)) {
    const vec = await embed(build(title || 'none', text))
    cosines[name].push(dot(vec, row))
    if (name !== 'document') continue
    const own = dot(vec, row)
    let better = 0
    for (let r = 0; r < rows; r++) if (r !== row && dot(vec, r) > own) better++
    if (better === 0) top1++
    if (better < 5) top5++
  }
}

const summary = Object.fromEntries(
  Object.entries(cosines).map(([name, xs]) => [
    name,
    { median: quantile(xs, 0.5), p10: quantile(xs, 0.1), min: Math.min(...xs) },
  ]),
)
// A doubled prefix moves the cosine by under 0.01, which is below the q4 noise, so bestPrompt only explains a failure.
const best = Object.entries(summary).sort((a, b) => b[1].median - a[1].median)[0][0]
const pass = summary.document.median >= minMedian && top1 / count >= 0.95
const report = {
  command:
    `pnpm exec tsx quartz/scripts/verify-semantic-index.ts ${process.argv.slice(2).join(' ')}`.trim(),
  model: manifest.model,
  onnxRepo,
  queryDtype: dtype,
  dims,
  rows,
  shardsVerified: manifest.vectors.shards.length,
  rowNorm: { min: minNorm, max: maxNorm },
  sampled: count,
  cosine: summary,
  bestPrompt: best,
  selfRetrieval: { top1: top1 / count, top5: top5 / count },
  minMedian,
  pass,
}
if (values.report) await writeFile(values.report, `${JSON.stringify(report, null, 2)}\n`)
console.log(JSON.stringify(report, null, 2))
process.exit(pass ? 0 : 1)
