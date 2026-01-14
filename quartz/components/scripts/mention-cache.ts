import { getDB, isDBAvailable, githubUsers, type GithubUser } from "./db"
import { ilike, or, desc } from "drizzle-orm"

const SYNC_KEY = "mentions-last-sync"
const SYNC_INTERVAL = 5 * 60 * 1000
const MAX_RETRIES = 3

let syncPromise: Promise<void> | null = null

async function syncFromD1(retries = MAX_RETRIES): Promise<void> {
  if (!isDBAvailable()) return

  const lastSync = parseInt(localStorage.getItem(SYNC_KEY) || "0")
  if (Date.now() - lastSync < SYNC_INTERVAL) return

  try {
    const db = await getDB()
    if (!db) return

    const resp = await fetch("/api/mentions")
    if (!resp.ok) throw new Error(`sync failed: ${resp.status}`)

    const users = (await resp.json()) as GithubUser[]

    await db.delete(githubUsers)
    if (users.length > 0) {
      await db.insert(githubUsers).values(users)
    }

    localStorage.setItem(SYNC_KEY, Date.now().toString())
  } catch (err) {
    if (retries > 0) {
      const backoff = (MAX_RETRIES - retries + 1) * 1000
      await new Promise((r) => setTimeout(r, backoff))
      return syncFromD1(retries - 1)
    }
    console.warn("mentions sync failed after retries:", err)
  }
}

export async function queryMentionUsers(query: string): Promise<GithubUser[]> {
  if (!syncPromise) {
    syncPromise = syncFromD1().finally(() => {
      syncPromise = null
    })
  }
  await syncPromise

  const db = await getDB()
  if (!db) return []

  const pattern = `%${query}%`
  return db
    .select()
    .from(githubUsers)
    .where(or(ilike(githubUsers.login, pattern), ilike(githubUsers.displayName, pattern)))
    .orderBy(desc(githubUsers.lastSeenAt))
    .limit(10)
}

export { isDBAvailable }
