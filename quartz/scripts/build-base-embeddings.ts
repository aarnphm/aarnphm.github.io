import { AutoConfig, AutoModel, AutoTokenizer } from '@huggingface/transformers'
// Builds one exact-scan vector index per base from the JSONL the BasePage emitter writes
// (public/<base>.embeddings-text.jsonl). It uses the ONNX encoder the browser loads, so the
// document and query vectors share one space without the MLX server. Output lands in
// content/embeddings/bases/<base>/, which the site build copies to /embeddings/bases/<base>/.
import { createHash } from 'node:crypto'
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { baseEmbeddingsDir, type BaseEmbeddingRecord } from '../util/base/embeddings'
import { SEMANTIC_ONNX_MODELS, type SemanticQueryDType } from '../util/semantic-search'

const JSONL_SUFFIX = '.embeddings-text.jsonl'

const { values } = parseArgs({
  options: {
    base: { type: 'string', multiple: true },
    'jsonl-dir': { type: 'string', default: 'public' },
    out: { type: 'string', default: 'content/embeddings/bases' },
    model: { type: 'string', default: 'google/embeddinggemma-2' },
    dims: { type: 'string', default: '768' },
    dtype: { type: 'string', default: 'q4' },
    'batch-size': { type: 'string', default: '8' },
    'shard-size': { type: 'string', default: '1024' },
    probe: { type: 'string', multiple: true },
    report: { type: 'string' },
  },
})

const modelId = values.model!
const dims = Number(values.dims)
const dtype = values.dtype as SemanticQueryDType
const batchSize = Math.max(1, Number(values['batch-size']))
const shardSize = Math.max(1, Number(values['shard-size']))
if (!modelId.toLowerCase().includes('embeddinggemma')) {
  throw new Error(`document prompts are only implemented for embeddinggemma, got ${modelId}`)
}
const onnxRepo = SEMANTIC_ONNX_MODELS[modelId]
if (!onnxRepo) throw new Error(`no ONNX repo mapped for ${modelId}`)

const jsonlDir = values['jsonl-dir']!
const available = (await readdir(jsonlDir))
  .filter(name => name.endsWith(JSONL_SUFFIX))
  .map(name => name.slice(0, -JSONL_SUFFIX.length))
  .sort()
const requested = values.base?.length ? values.base : available
const missing = requested.filter(slug => !available.includes(slug))
if (missing.length > 0) {
  throw new Error(
    `no ${JSONL_SUFFIX} for ${missing.join(', ')} in ${jsonlDir}; run the site build first`,
  )
}
if (requested.length === 0) throw new Error(`no ${JSONL_SUFFIX} files in ${jsonlDir}`)

const config = await AutoConfig.from_pretrained(onnxRepo)
if (config.model_type === 'embedding_gemma2') {
  Object.assign(config, { vision_config: null, audio_config: null })
}
const tokenizer = await AutoTokenizer.from_pretrained(onnxRepo)
const model = await AutoModel.from_pretrained(onnxRepo, { config, dtype })

async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const outputs = await model(tokenizer(texts, { padding: true, truncation: true }))
  const data = outputs.sentence_embedding.data as Float32Array
  const width = outputs.sentence_embedding.dims.at(-1) as number
  if (width < dims) throw new Error(`model emits ${width} dims but ${dims} were requested`)
  return texts.map((_, i) => {
    // Matryoshka rule: keep the leading dims, then renormalize.
    const vec = Float32Array.from(data.subarray(i * width, i * width + dims))
    const norm = Math.hypot(...vec)
    if (!Number.isFinite(norm) || norm === 0) throw new Error(`row ${i} has a degenerate vector`)
    return vec.map(v => v / norm)
  })
}

const documentPrompt = ({ title, text }: BaseEmbeddingRecord) =>
  `title: ${title || 'none'} | text: ${text}`
const queryPrompt = (query: string) => `task: search result | query: ${query}`

type ShardMeta = {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  sha256: string
  byteStride: number
}

async function buildBase(slug: string) {
  const started = performance.now()
  const rows: BaseEmbeddingRecord[] = []
  for (const line of (await readFile(path.join(jsonlDir, `${slug}${JSONL_SUFFIX}`), 'utf8')).split(
    '\n',
  )) {
    if (line.trim()) rows.push(JSON.parse(line) as BaseEmbeddingRecord)
  }
  if (rows.length === 0) throw new Error(`${slug}: the JSONL holds no rows`)

  const vectors = new Float32Array(rows.length * dims)
  for (let start = 0; start < rows.length; start += batchSize) {
    const batch = rows.slice(start, start + batchSize)
    const embedded = await embedBatch(batch.map(documentPrompt))
    embedded.forEach((vec, i) => vectors.set(vec, (start + i) * dims))
    process.stdout.write(`\r${slug}: ${Math.min(start + batchSize, rows.length)}/${rows.length}`)
  }
  process.stdout.write('\n')

  const outDir = path.join(values.out!, slug)
  await mkdir(outDir, { recursive: true })
  for (const name of await readdir(outDir)) {
    if (/^vectors-\d+\.bin$/.test(name) || name === 'manifest.json') {
      await rm(path.join(outDir, name))
    }
  }
  const shards: ShardMeta[] = []
  for (let offset = 0, index = 0; offset < rows.length; offset += shardSize, index++) {
    const count = Math.min(shardSize, rows.length - offset)
    const view = vectors.subarray(offset * dims, (offset + count) * dims)
    const payload = Buffer.from(view.buffer, view.byteOffset, view.byteLength)
    const name = `vectors-${String(index).padStart(3, '0')}.bin`
    await writeFile(path.join(outDir, name), payload)
    shards.push({
      path: `/${baseEmbeddingsDir}/${slug}/${name}`,
      rows: count,
      rowOffset: offset,
      byteLength: payload.byteLength,
      sha256: createHash('sha256').update(payload).digest('hex'),
      byteStride: dims * 4,
    })
  }
  const manifest = {
    version: 2,
    model: modelId,
    dims,
    dtype: 'fp32',
    normalized: true,
    rows: rows.length,
    shardSizeRows: shardSize,
    vectors: { dtype: 'fp32', rows: rows.length, dims, shards },
    ids: rows.map(row => row.slug),
    titles: rows.map(row => row.title),
    encoder: { repo: onnxRepo, dtype },
  }
  await writeFile(path.join(outDir, 'manifest.json'), JSON.stringify(manifest))

  const probes: Record<string, { slug: string; title: string; score: number }[]> = {}
  for (const query of values.probe ?? []) {
    const [q] = await embedBatch([queryPrompt(query)])
    const scored = rows.map((row, r) => {
      let score = 0
      for (let d = 0; d < dims; d++) score += q[d] * vectors[r * dims + d]
      return { slug: row.slug, title: row.title, score }
    })
    probes[query] = scored.sort((a, b) => b.score - a.score).slice(0, 5)
    console.log(`${slug} ← "${query}"`)
    for (const hit of probes[query]) console.log(`  ${hit.score.toFixed(3)}  ${hit.title}`)
  }
  return {
    base: slug,
    rows: rows.length,
    shards: shards.length,
    bytes: shards.reduce((sum, shard) => sum + shard.byteLength, 0),
    elapsedMs: Math.round(performance.now() - started),
    probes,
  }
}

const results = []
for (const slug of requested) results.push(await buildBase(slug))
const report = {
  command:
    `pnpm exec tsx quartz/scripts/build-base-embeddings.ts ${process.argv.slice(2).join(' ')}`.trim(),
  model: modelId,
  onnxRepo,
  encoderDtype: dtype,
  dims,
  bases: results,
}
if (values.report) await writeFile(values.report, `${JSON.stringify(report, null, 2)}\n`)
for (const r of results) {
  console.log(`${r.base}: ${r.rows} rows, ${(r.bytes / 1024).toFixed(0)} KB, ${r.elapsedMs} ms`)
}
