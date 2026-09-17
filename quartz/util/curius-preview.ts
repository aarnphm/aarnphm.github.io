import { z } from 'zod'

const publicUrl = z
  .string()
  .max(8192)
  .refine(value => {
    const url = URL.parse(value)
    return Boolean(
      url && ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password,
    )
  })

const preview = z.discriminatedUnion('status', [
  z.object({
    status: z.literal('ready'),
    linkId: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
    title: z.string().max(4096),
    sourceUrl: publicUrl,
    finalUrl: publicUrl,
    readerHtml: z
      .string()
      .min(1)
      .max(2 * 1024 * 1024),
    fetchedAt: z.number().int().nonnegative(),
    cached: z.boolean(),
  }),
  z.object({
    status: z.literal('unavailable'),
    reason: z.string().min(1).max(128),
    message: z.string().min(1).max(4096),
    sourceUrl: publicUrl.optional(),
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
