import { z } from 'zod'

export const QUARTZ_CACHE_RECEIPT_PREFIX = '__QUARTZ_CACHE_RECEIPT__'

const receiptSchema = z.object({
  epoch: z.string(),
  profile: z.enum(['watch', 'production', 'build']),
  phase: z.enum(['inputs', 'restore', 'admission']),
  status: z.enum(['disabled', 'miss', 'parse-hit', 'output-hit', 'saved', 'skipped']),
  reason: z.enum([
    'non-watch',
    'forced',
    'missing-parser-inputs',
    'inputs-unavailable',
    'missing-manifest',
    'incompatible-manifest',
    'corrupt-manifest',
    'parse-inputs-changed',
    'missing-snapshot',
    'corrupt-snapshot',
    'build-inputs-changed',
    'output-changed',
    'unchanged',
    'unsupported-metadata',
    'serialization-failed',
    'missing-snapshot-result',
    'inputs-unstable',
    'unavailable',
    'published',
    'parse-unchanged',
    'resources-updated',
    'checkpoint-changed',
  ]),
  manifestPath: z.string(),
  detail: z.string().optional(),
  counts: z
    .object({
      parsed: z.number().int().nonnegative(),
      rendered: z.number().int().nonnegative(),
      outputWrites: z.number().int().nonnegative(),
    })
    .optional(),
})

export type QuartzCacheReceipt = z.infer<typeof receiptSchema>

export function emitQuartzCacheReceipt(receipt: QuartzCacheReceipt): void {
  console.log(`${QUARTZ_CACHE_RECEIPT_PREFIX}${JSON.stringify(receipt)}`)
}

export function parseQuartzCacheReceipt(line: string): QuartzCacheReceipt | undefined {
  const start = line.indexOf(QUARTZ_CACHE_RECEIPT_PREFIX)
  if (start === -1) return undefined
  try {
    const parsed: unknown = JSON.parse(line.slice(start + QUARTZ_CACHE_RECEIPT_PREFIX.length))
    const receipt = receiptSchema.safeParse(parsed)
    return receipt.success ? receipt.data : undefined
  } catch {
    return undefined
  }
}

export function formatQuartzCacheReceipt(receipt: QuartzCacheReceipt): string {
  const counts = receipt.counts
  const work = counts
    ? `; ${counts.parsed} parsed, ${counts.rendered} rendered, ${counts.outputWrites} output writes`
    : ''
  return `cache ${receipt.phase}: ${receipt.status} (${receipt.reason})${work}; manifest ${receipt.manifestPath}${receipt.detail ? `; ${receipt.detail}` : ''}`
}
