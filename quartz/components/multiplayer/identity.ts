import { blobDataUrl, decodeBlob } from '../../util/blob-avatar'
import { getFullSlug } from '../../util/path'
import { isRecord } from './model'

export function getAuthor(): string {
  const login = localStorage.getItem('comment-author-github-login')
  const stored = localStorage.getItem('comment-author')
  if (login && stored !== login) {
    localStorage.setItem('comment-author', login)
    return login
  }
  if (stored) return stored
  if (login) {
    localStorage.setItem('comment-author', login)
    return login
  }
  const author = `anon-${Math.random().toString(36).slice(2, 8)}`
  localStorage.setItem('comment-author', author)
  return author
}

export function getCommentPageId(): string {
  const slug = getFullSlug(window)
  const hostname = window.location.hostname.toLowerCase()
  if (hostname === 'stream.aarnphm.xyz') {
    return `stream:${slug}`
  }
  return slug
}

// Avatars are blobs (quartz/util/blob-avatar.ts): the name picks one, and a commenter who pinned
// traits from the palette has them stored as a code that every reader fetches. Lookups made in the
// same tick share one request.
export const ownAvatarKey = 'comment-avatar'
const avatarCodes = new Map<string, string>()
let pending = new Map<string, ((code: string) => void)[]>()
let flush: Promise<void> | null = null

const fetchCodes = async () => {
  await Promise.resolve()
  const batch = pending
  pending = new Map()
  flush = null
  const url = new URL('/comments/avatars', window.location.origin)
  for (const author of batch.keys()) url.searchParams.append('author', author)
  let avatars: Record<string, unknown> = {}
  try {
    const resp = await fetch(url)
    const data: unknown = resp.ok ? await resp.json() : null
    if (isRecord(data) && isRecord(data['avatars'])) avatars = data['avatars']
  } catch {}
  for (const [author, waiting] of batch) {
    const code = typeof avatars[author] === 'string' ? avatars[author] : ''
    avatarCodes.set(author, code)
    for (const resolve of waiting) resolve(code)
  }
}

export function getAvatarCode(author: string): Promise<string> {
  if (author === localStorage.getItem('comment-author')) {
    const own = localStorage.getItem(ownAvatarKey)
    if (own !== null) return Promise.resolve(own)
  }
  const known = avatarCodes.get(author)
  if (known !== undefined) return Promise.resolve(known)
  return new Promise(resolve => {
    pending.set(author, [...(pending.get(author) ?? []), resolve])
    flush ??= fetchCodes()
  })
}

export async function getAvatarUrl(author: string): Promise<string> {
  return blobDataUrl(author, decodeBlob(await getAvatarCode(author)))
}

// Saves this commenter's pinned traits and redraws their avatars already on the page.
export async function setOwnAvatar(code: string): Promise<boolean> {
  const author = getAuthor()
  const resp = await fetch('/comments/avatars', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ author, code }),
  }).catch(() => null)
  if (!resp?.ok) return false
  localStorage.setItem(ownAvatarKey, code)
  avatarCodes.set(author, code)
  const src = blobDataUrl(author, decodeBlob(code))
  for (const img of document.querySelectorAll<HTMLImageElement>('img.reply-avatar, img.avatar'))
    if (img.alt === author) img.src = src
  return true
}
