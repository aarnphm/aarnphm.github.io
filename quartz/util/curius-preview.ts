// zod/mini keeps the client chunk small: the full zod build ships about 440 KB of minified code.
import * as z from 'zod/mini'

const publicUrl = z.string().check(
  z.maxLength(8192),
  z.refine(value => {
    const url = URL.parse(value)
    return Boolean(
      url && ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password,
    )
  }),
)

const preview = z.discriminatedUnion('status', [
  z.object({
    status: z.literal('ready'),
    linkId: z.int().check(z.positive(), z.maximum(Number.MAX_SAFE_INTEGER)),
    title: z.string().check(z.maxLength(4096)),
    sourceUrl: publicUrl,
    finalUrl: publicUrl,
    readerHtml: z.string().check(z.minLength(1), z.maxLength(2 * 1024 * 1024)),
    fetchedAt: z.int().check(z.nonnegative()),
    cached: z.boolean(),
  }),
  z.object({
    status: z.literal('unavailable'),
    reason: z.string().check(z.minLength(1), z.maxLength(128)),
    message: z.string().check(z.minLength(1), z.maxLength(4096)),
    sourceUrl: z.optional(publicUrl),
  }),
])

export type CuriusPreviewResponse = z.infer<typeof preview>

export function parseCuriusPreview(value: unknown): CuriusPreviewResponse | null {
  const parsed = preview.safeParse(value)
  return parsed.success ? parsed.data : null
}

export function curiusPreviewImagePath(linkId: number, index: number, fetchedAt: number): string {
  return `/api/curius?query=preview-image&id=${linkId}&image=${index}&v=${fetchedAt}`
}

export function isCuriusPreviewImageUrl(raw: string, origin: string): boolean {
  const url = URL.parse(raw, origin)
  if (
    !url ||
    url.origin !== origin ||
    url.pathname !== '/api/curius' ||
    url.username ||
    url.password ||
    url.hash ||
    url.searchParams.size !== 4 ||
    url.searchParams.get('query') !== 'preview-image'
  )
    return false
  const id = url.searchParams.get('id') ?? ''
  const image = url.searchParams.get('image') ?? ''
  const version = url.searchParams.get('v') ?? ''
  return (
    /^[1-9]\d*$/.test(id) &&
    Number.isSafeInteger(Number(id)) &&
    /^(0|[1-9]\d*)$/.test(image) &&
    Number(image) < 300 &&
    /^[1-9]\d*$/.test(version) &&
    Number.isSafeInteger(Number(version))
  )
}
