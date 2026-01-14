import { getFullSlug } from "../../util/path"
import { isRecord } from "./model"

const githubAvatarCache = new Map<string, string>()
const githubAvatarStoragePrefix = "comment-author-github-avatar:"

export function getAuthor(): string {
  let author =
    localStorage.getItem("comment-author") ?? localStorage.getItem("comment-author-github-login")
  if (!author) {
    author = `anon-${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem("comment-author", author)
  }
  return author
}

export function getCommentPageId(): string {
  const slug = getFullSlug(window)
  const hostname = window.location.hostname.toLowerCase()
  if (hostname === "stream.aarnphm.xyz") {
    return `stream:${slug}`
  }
  return slug
}

export async function getGravatarUrl(identifier: string, size: number = 24): Promise<string> {
  const normalized = identifier.trim().toLowerCase()
  const encoder = new TextEncoder()
  const data = encoder.encode(normalized)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
  return `https://gravatar.com/avatar/${hashHex}?s=${size}&d=identicon&r=pg`
}

export async function getGithubAvatarUrl(login: string): Promise<string | null> {
  const cached = githubAvatarCache.get(login)
  if (cached) return cached
  const storageKey = `${githubAvatarStoragePrefix}${login}`
  try {
    const stored = sessionStorage.getItem(storageKey)
    if (stored) {
      githubAvatarCache.set(login, stored)
      return stored
    }
  } catch {}
  const resp = await fetch(`https://api.github.com/users/${encodeURIComponent(login)}`, {
    headers: { Accept: "application/vnd.github+json" },
  })
  if (!resp.ok) return null
  let data: unknown
  try {
    data = await resp.json()
  } catch {
    return null
  }
  if (!isRecord(data)) return null
  const avatar = data["avatar_url"]
  if (typeof avatar !== "string" || avatar.length === 0) return null
  githubAvatarCache.set(login, avatar)
  try {
    sessionStorage.setItem(storageKey, avatar)
  } catch {}
  return avatar
}

export async function getAvatarUrl(author: string, size: number = 24): Promise<string> {
  const login = localStorage.getItem("comment-author-github-login")
  const localAuthor = localStorage.getItem("comment-author")
  if (login && (author === localAuthor || author === login)) {
    const githubUrl = await getGithubAvatarUrl(login)
    if (githubUrl) return githubUrl
  }
  return getGravatarUrl(author, size)
}
