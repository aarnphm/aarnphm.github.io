import { parsePdfMarkInput, type PdfMarkInput } from '../../util/pdf-marks'

const databaseName = 'pdf-reader-v1'

/** A write the server has not acknowledged yet; it survives reloads and retries when online. */
export type PendingOp =
  | { id: string; src: string; type: 'put'; input: PdfMarkInput; queuedAt: number }
  | { id: string; src: string; type: 'delete'; revision: number; queuedAt: number }

export interface ReadingPosition {
  page: number
  top: number
}

let opened: Promise<IDBDatabase> | undefined

function open(): Promise<IDBDatabase> {
  opened ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1)
    request.onupgradeneeded = () => {
      const database = request.result
      database.createObjectStore('pending', { keyPath: 'id' }).createIndex('src', 'src')
      database.createObjectStore('positions')
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Local mark storage is unavailable.'))
    request.onblocked = () => reject(new Error('Close older reader tabs to enable local storage.'))
  })
  opened.catch(() => {
    opened = undefined
  })
  return opened
}

function done(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
    transaction.onabort = () => reject(transaction.error)
  })
}

function readPending(value: unknown): PendingOp | null {
  if (typeof value !== 'object' || value === null) return null
  const record = value as Record<string, unknown>
  if (typeof record.id !== 'string' || typeof record.src !== 'string') return null
  const queuedAt = typeof record.queuedAt === 'number' ? record.queuedAt : 0
  if (record.type === 'put') {
    const input = parsePdfMarkInput(record.input)
    return input ? { id: record.id, src: record.src, type: 'put', input, queuedAt } : null
  }
  if (record.type === 'delete' && typeof record.revision === 'number') {
    return { id: record.id, src: record.src, type: 'delete', revision: record.revision, queuedAt }
  }
  return null
}

export async function loadPending(src: string): Promise<PendingOp[]> {
  const database = await open()
  return new Promise((resolve, reject) => {
    const request = database.transaction('pending').objectStore('pending').index('src').getAll(src)
    request.onsuccess = () => {
      const values: unknown = request.result
      resolve(
        (Array.isArray(values) ? values.map(readPending) : [])
          .filter((op): op is PendingOp => op !== null)
          .sort((a, b) => a.queuedAt - b.queuedAt),
      )
    }
    request.onerror = () => reject(request.error)
  })
}

export async function savePending(op: PendingOp): Promise<void> {
  const database = await open()
  const transaction = database.transaction('pending', 'readwrite')
  transaction.objectStore('pending').put(op)
  return done(transaction)
}

export async function removePending(id: string): Promise<void> {
  const database = await open()
  const transaction = database.transaction('pending', 'readwrite')
  transaction.objectStore('pending').delete(id)
  return done(transaction)
}

export async function savePosition(doc: string, position: ReadingPosition): Promise<void> {
  const database = await open()
  const transaction = database.transaction('positions', 'readwrite')
  transaction.objectStore('positions').put(position, doc)
  return done(transaction)
}

export async function loadPosition(doc: string): Promise<ReadingPosition | null> {
  const database = await open()
  return new Promise((resolve, reject) => {
    const request = database.transaction('positions').objectStore('positions').get(doc)
    request.onsuccess = () => {
      const value: unknown = request.result
      if (typeof value !== 'object' || value === null) return resolve(null)
      const { page, top } = value as Record<string, unknown>
      resolve(
        typeof page === 'number' && Number.isInteger(page) && page >= 1 && typeof top === 'number'
          ? { page, top: Math.min(1, Math.max(0, top)) }
          : null,
      )
    }
    request.onerror = () => reject(request.error)
  })
}
