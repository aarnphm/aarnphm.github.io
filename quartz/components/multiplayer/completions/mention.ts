import { CompletionContext, CompletionResult } from '@codemirror/autocomplete'
import { desc } from 'drizzle-orm'
import { getDB, githubUsers, type GithubUser } from '../../scripts/db'
import { fuzzyMatchMultiple } from './fuzzy'

const SYNC_KEY = 'mentions-last-sync'
const SYNC_INTERVAL = 5 * 60 * 1000

let syncPromise: Promise<void> | null = null

async function syncFromD1() {
  const lastSync = parseInt(localStorage.getItem(SYNC_KEY) || '0')
  if (Date.now() - lastSync < SYNC_INTERVAL) return

  try {
    const db = await getDB()
    if (!db) return

    const resp = await fetch('/api/mentions')
    if (!resp.ok) throw new Error(resp.statusText)

    const users = (await resp.json()) as GithubUser[]
    await db.delete(githubUsers)
    if (users.length > 0) {
      await db.insert(githubUsers).values(users)
    }

    localStorage.setItem(SYNC_KEY, Date.now().toString())
  } catch (err) {
    console.warn('mentions sync failed:', err)
  }
}

export async function mentionCompletionSource(
  context: CompletionContext,
): Promise<CompletionResult | null> {
  const word = context.matchBefore(/@[\w-]*/)
  if (!word || (word.from === word.to && !context.explicit)) return null

  const query = word.text.slice(1)
  if (!syncPromise) {
    syncPromise = syncFromD1().finally(() => {
      syncPromise = null
    })
  }
  await syncPromise

  const db = await getDB()
  if (!db) return null

  const allUsers = await db.select().from(githubUsers).orderBy(desc(githubUsers.lastSeenAt))
  const scored = []

  for (const user of allUsers) {
    if (query.length === 0) {
      scored.push({ user, score: 0 })
      continue
    }

    const targets = [user.login]
    if (user.displayName && user.displayName !== user.login) targets.push(user.displayName)

    const best = fuzzyMatchMultiple(query, targets)
    if (best) {
      scored.push({ user, score: best.match.score })
    }
  }

  if (query.length > 0) {
    scored.sort((a, b) => b.score - a.score)
  }

  const limited = scored.slice(0, 10)
  if (limited.length === 0) return null

  return {
    from: word.from,
    filter: false,
    options: limited.map(({ user, score }) => ({
      label: `@${user.login}`,
      detail: user.displayName && user.displayName !== user.login ? user.displayName : undefined,
      type: 'mention',
      apply: `@${user.login} `,
      boost: score,
    })),
  }
}
