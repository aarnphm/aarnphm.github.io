const DOT_ESCAPE = '___DOT___'
export const STACKED_NOTE_METADATA_CLASSES = ['modified-time', 'published-time', 'reading-time']

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (const byte of bytes) {
    binary += String.fromCharCode(byte)
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function base64ToBytes(hash: string): Uint8Array | null {
  const base64 = hash.replace(/-/g, '+').replace(/_/g, '/')
  const padding = (4 - (base64.length % 4)) % 4
  let binary: string
  try {
    binary = atob(`${base64}${'='.repeat(padding)}`)
  } catch {
    return null
  }
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

export function normalizeStackedNoteSlug(raw: string | null): string | null {
  if (!raw) return null
  let decoded: string
  try {
    decoded = decodeURIComponent(raw)
  } catch {
    return null
  }
  const slug = decoded.replace(/^\/+|\/+$/g, '')
  if (!slug) return 'index'
  if (slug.split('/').some(part => part === '' || part === '.' || part === '..')) return null
  if (slug.includes('?') || slug.includes('#') || slug.includes('\\')) return null
  if ([...slug].some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) return null
  return slug
}

export function hashStackedNoteSlug(slug: string): string {
  const safePath = slug.toString().replace(/\./g, DOT_ESCAPE)
  return bytesToBase64(new TextEncoder().encode(safePath))
}

export function decodeStackedNoteHash(hash: string): string | null {
  const bytes = base64ToBytes(hash)
  if (!bytes) return null

  let decoded: string
  try {
    decoded = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    return null
  }

  return normalizeStackedNoteSlug(decoded.replace(/___DOT___/g, '.'))
}

function fragmentHasClass(fragment: string, className: string): boolean {
  return (
    fragment.includes(`"${className}`) ||
    fragment.includes(` ${className}`) ||
    fragment.includes(`${className}"`)
  )
}

export function stackedNoteMetadataHtml(items: string[]): string {
  const ordered: string[] = []
  for (const className of STACKED_NOTE_METADATA_CLASSES) {
    for (const item of items) {
      if (!fragmentHasClass(item, className) || ordered.includes(item)) continue
      ordered.push(item)
    }
  }

  if (ordered.length === 0) return ''

  return `<footer class="stacked-note-footer" aria-label="note metadata">
  <ul class="content-meta stacked-note-content-meta">
${ordered.map(item => `    ${item}`).join('\n')}
  </ul>
</footer>`
}
