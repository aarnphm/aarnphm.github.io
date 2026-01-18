import type { FrecencyEntry } from "./types"

const KEY = "completion-frecency"
const HALF_LIFE = 24 * 60 * 60 * 1000
const MAX = 500

class FrecencyStore {
  data: Record<string, FrecencyEntry> = {}

  constructor() {
    try {
      this.data = JSON.parse(localStorage.getItem(KEY) || "{}")
    } catch {}
  }

  score(entry: FrecencyEntry, now = Date.now()) {
    return Math.log2(entry.count + 1) * Math.pow(0.5, (now - entry.lastAccess) / HALF_LIFE)
  }

  save() {
    try {
      const entries = Object.entries(this.data)
      if (entries.length > MAX) {
        const now = Date.now()
        entries.sort(([, a], [, b]) => this.score(b, now) - this.score(a, now))
        this.data = Object.fromEntries(entries.slice(0, MAX))
      }
      localStorage.setItem(KEY, JSON.stringify(this.data))
    } catch {}
  }

  recordAccess(key: string) {
    const { count = 0 } = this.data[key] ?? {}
    this.data[key] = { count: count + 1, lastAccess: Date.now() }
    this.save()
  }

  getBoost(key: string) {
    return this.data[key] ? this.score(this.data[key]) * 10 : 0
  }

  getEntry(key: string) {
    return this.data[key]
  }
}

export const frecencyStore = new FrecencyStore()
