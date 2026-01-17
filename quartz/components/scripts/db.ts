import { PGlite } from "@electric-sql/pglite"
import { drizzle, type PgliteDatabase } from "drizzle-orm/pglite"
import { pgTable, text, integer, index } from "drizzle-orm/pg-core"
import { dependencies } from "../../../package.json"

const CDN_BASE = `https://cdn.jsdelivr.net/npm/@electric-sql/pglite@${dependencies["@electric-sql/pglite"].slice(1)}/dist`

export const githubUsers = pgTable(
  "github_users",
  {
    login: text("login").primaryKey(),
    displayName: text("display_name"),
    avatarUrl: text("avatar_url"),
    lastSeenAt: integer("last_seen_at").notNull(),
  },
  (table) => [index("idx_github_users_last_seen").on(table.lastSeenAt)],
)

export type GithubUser = typeof githubUsers.$inferSelect

let client: PGlite | null = null
let db: PgliteDatabase | null = null
let initPromise: Promise<PgliteDatabase | null> | null = null
let initFailed = false

async function initDB(): Promise<PgliteDatabase | null> {
  if (initFailed) return null
  if (db) return db

  try {
    const [wasmModule, fsBundle] = await Promise.all([
      WebAssembly.compileStreaming(fetch(`${CDN_BASE}/pglite.wasm`)),
      fetch(`${CDN_BASE}/pglite.data`).then((r) => r.blob()),
    ])

    client = await PGlite.create({
      dataDir: "idb://multiplayer-cache",
      wasmModule,
      fsBundle,
    })
    db = drizzle({ client })

    await client.exec(`
      CREATE TABLE IF NOT EXISTS github_users (
        login TEXT PRIMARY KEY,
        display_name TEXT,
        avatar_url TEXT,
        last_seen_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_github_users_last_seen
        ON github_users(last_seen_at DESC);
    `)

    return db
  } catch (err) {
    console.warn("pglite init failed:", err)
    initFailed = true
    return null
  }
}

export async function getDB(): Promise<PgliteDatabase | null> {
  if (initFailed) return null
  if (db) return db
  if (!initPromise) {
    initPromise = initDB()
  }
  return initPromise
}

export function isDBAvailable(): boolean {
  return !initFailed
}

export async function getClient(): Promise<PGlite | null> {
  await getDB()
  return client
}
