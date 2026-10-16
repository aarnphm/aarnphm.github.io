import type { PreparedForegroundTree, TreeWorkerResponse } from './tree-worker-types'
import { isRecord } from '../../util/type-guards'

export type CachedSceneTrees = { trees: TreeWorkerResponse; foreground: PreparedForegroundTree[] }

const databaseName = 'garden-trail-trees'
const storeName = 'scenes'
const maximumBytes = 8 * 1024 * 1024
const maximumScenes = 3
let database: Promise<IDBDatabase | null> | null = null

function openCache(): Promise<IDBDatabase | null> {
  database ??= new Promise(resolve => {
    let unavailable = false
    try {
      const request = indexedDB.open(databaseName, 1)
      request.onupgradeneeded = () => {
        const store = request.result.createObjectStore(storeName, { keyPath: 'key' })
        store.createIndex('written', 'written')
      }
      request.onerror = request.onblocked = () => {
        unavailable = true
        resolve(null)
      }
      request.onsuccess = () => {
        const db = request.result
        if (unavailable) {
          db.close()
          return
        }
        db.onversionchange = () => {
          db.close()
          database = null
        }
        resolve(db)
      }
    } catch {
      resolve(null)
    }
  })
  return database
}

function isCachedScene(value: unknown): value is CachedSceneTrees {
  return (
    isRecord(value) &&
    Array.isArray(value.trees) &&
    value.treeCount === value.trees.length &&
    value.trees.every(
      item =>
        isRecord(item) &&
        typeof item.id === 'string' &&
        typeof item.markup === 'string' &&
        item.markup.startsWith('<g ') &&
        item.markup.includes(`data-tree-id="${item.id}"`),
    ) &&
    Array.isArray(value.foreground) &&
    value.foregroundCount === value.foreground.length &&
    new Set(value.foreground.map(item => (isRecord(item) ? item.id : null))).size ===
      value.foreground.length &&
    value.foreground.every(
      item =>
        isRecord(item) &&
        typeof item.id === 'string' &&
        typeof item.bottom === 'number' &&
        Number.isFinite(item.bottom) &&
        typeof item.shape === 'string' &&
        typeof item.markup === 'string' &&
        item.markup.startsWith('<g ') &&
        item.markup.includes(`data-tree-id="${item.id}"`),
    )
  )
}

export async function sceneTreeCacheKey(inputs: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(inputs))
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

export async function readSceneTrees(key: string): Promise<CachedSceneTrees | null> {
  const db = await openCache()
  if (!db) return null
  return new Promise(resolve => {
    try {
      const transaction = db.transaction(storeName, 'readonly')
      const request = transaction.objectStore(storeName).get(key)
      request.onerror = transaction.onabort = () => resolve(null)
      request.onsuccess = () => {
        const value: unknown = request.result
        resolve(isCachedScene(value) ? value : null)
      }
    } catch {
      resolve(null)
    }
  })
}

export async function writeSceneTrees(key: string, scene: CachedSceneTrees): Promise<void> {
  const bytes =
    2 *
    (scene.trees.reduce((sum, item) => sum + item.markup.length + item.id.length, 0) +
      scene.foreground.reduce(
        (sum, item) => sum + item.markup.length + item.shape.length + item.id.length,
        0,
      ))
  if (bytes > maximumBytes) return
  const db = await openCache()
  if (!db) return
  return new Promise(resolve => {
    try {
      const transaction = db.transaction(storeName, 'readwrite')
      transaction.oncomplete = transaction.onabort = transaction.onerror = () => resolve()
      const store = transaction.objectStore(storeName)
      store.put({
        key,
        written: Date.now(),
        trees: scene.trees,
        foreground: scene.foreground,
        treeCount: scene.trees.length,
        foregroundCount: scene.foreground.length,
      })
      // Keep only three complete forests, without reading their large SVG values to evict them.
      const request = store.index('written').openKeyCursor(null, 'prev')
      let count = 0
      request.onsuccess = () => {
        const cursor = request.result
        if (!cursor) return
        if (++count > maximumScenes) store.delete(cursor.primaryKey)
        cursor.continue()
      }
    } catch {
      resolve()
    }
  })
}
