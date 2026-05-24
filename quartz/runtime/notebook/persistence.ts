import type { CellId } from '../../util/notebook/types'

const STORAGE_PREFIX = 'quartz:nb:v2:'
const LEGACY_PREFIX = 'quartz:notebook-source:'

function storageKey(sourcePath: string, cellId: CellId): string {
  return `${STORAGE_PREFIX}${encodeURIComponent(sourcePath)}:${encodeURIComponent(cellId)}`
}

function legacyKey(sourcePath: string, cellId: CellId): string {
  return `${LEGACY_PREFIX}${encodeURIComponent(sourcePath)}:${encodeURIComponent(cellId)}`
}

export type CellSourceStorage = {
  read(sourcePath: string, cellId: CellId): string | undefined
  write(sourcePath: string, cellId: CellId, value: string): void
  remove(sourcePath: string, cellId: CellId): void
  migrate(sourcePath: string, cellIds: readonly CellId[]): void
}

function pickStorage(): Storage | undefined {
  if (typeof globalThis === 'undefined') return undefined
  const localStorage = (globalThis as { localStorage?: Storage }).localStorage
  if (!localStorage) return undefined
  return localStorage
}

export function createCellSourceStorage(
  storage: Storage | undefined = pickStorage(),
): CellSourceStorage {
  if (!storage) {
    return {
      read: () => undefined,
      write: () => undefined,
      remove: () => undefined,
      migrate: () => undefined,
    }
  }
  return {
    read(sourcePath, cellId) {
      return storage.getItem(storageKey(sourcePath, cellId)) ?? undefined
    },
    write(sourcePath, cellId, value) {
      storage.setItem(storageKey(sourcePath, cellId), value)
    },
    remove(sourcePath, cellId) {
      storage.removeItem(storageKey(sourcePath, cellId))
    },
    migrate(sourcePath, cellIds) {
      for (const cellId of cellIds) {
        const key = storageKey(sourcePath, cellId)
        if (storage.getItem(key) !== null) continue
        const legacy = storage.getItem(legacyKey(sourcePath, cellId))
        if (legacy !== null) {
          storage.setItem(key, legacy)
          storage.removeItem(legacyKey(sourcePath, cellId))
        }
      }
    },
  }
}

export const cellSourceStoragePrefix = STORAGE_PREFIX
export const cellSourceLegacyPrefix = LEGACY_PREFIX
