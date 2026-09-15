import type { NoteDraft, ReaderPass } from './model'
import { isNote, isRecord } from './api'

const databaseName = 'arena-reader-v1'

export function openReaderStore(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1)
    request.onupgradeneeded = () => {
      const database = request.result
      const drafts = database.createObjectStore('drafts', { keyPath: ['subject', 'note.id'] })
      drafts.createIndex('subject', 'subject')
      database.createObjectStore('positions')
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Local note storage is unavailable.'))
    request.onblocked = () =>
      reject(new Error('Close older reader tabs to enable local note storage.'))
  })
}

function isDraft(value: unknown): value is NoteDraft {
  return (
    isRecord(value) &&
    typeof value.subject === 'string' &&
    isNote(value.note) &&
    typeof value.dirty === 'boolean' &&
    typeof value.ready === 'boolean' &&
    typeof value.localVersion === 'number' &&
    (value.conflict === null || isNote(value.conflict))
  )
}

export async function loadDrafts(
  database: Promise<IDBDatabase>,
  subject: string,
): Promise<NoteDraft[]> {
  const db = await database
  return new Promise((resolve, reject) => {
    const request = db.transaction('drafts').objectStore('drafts').index('subject').getAll(subject)
    request.onsuccess = () => {
      const values: unknown = request.result
      resolve(
        Array.isArray(values)
          ? values.filter(isDraft).filter(draft => draft.subject === subject)
          : [],
      )
    }
    request.onerror = () => reject(request.error)
  })
}

export async function saveDraft(database: Promise<IDBDatabase>, draft: NoteDraft): Promise<void> {
  const db = await database
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('drafts', 'readwrite')
    transaction.objectStore('drafts').put(draft)
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
    transaction.onabort = () => reject(transaction.error)
  })
}

export async function removeDraft(
  database: Promise<IDBDatabase>,
  subject: string,
  id: string,
): Promise<void> {
  const db = await database
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('drafts', 'readwrite')
    transaction.objectStore('drafts').delete([subject, id])
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
  })
}

export function loadPass(subject: string): ReaderPass {
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem(`${databaseName}:pass:${subject}`) ?? 'null',
    )
    if (
      isRecord(value) &&
      typeof value.seed === 'string' &&
      (value.current === null || typeof value.current === 'string') &&
      Array.isArray(value.visited) &&
      value.visited.every(id => typeof id === 'string')
    ) {
      return { seed: value.seed, current: value.current, visited: value.visited }
    }
  } catch {
    /* A fresh pass remains usable when browser storage is unavailable. */
  }
  return { seed: crypto.randomUUID(), current: null, visited: [] }
}

export function savePass(subject: string, pass: ReaderPass): void {
  localStorage.setItem(`${databaseName}:pass:${subject}`, JSON.stringify(pass))
}

export async function savePosition(
  database: Promise<IDBDatabase>,
  subject: string,
  articleId: string,
  fingerprint: string,
  fraction: number,
): Promise<void> {
  const db = await database
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('positions', 'readwrite')
    transaction.objectStore('positions').put({ fingerprint, fraction }, [subject, articleId])
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
  })
}

export async function loadPosition(
  database: Promise<IDBDatabase>,
  subject: string,
  articleId: string,
  fingerprint: string,
): Promise<number> {
  const db = await database
  return new Promise((resolve, reject) => {
    const request = db.transaction('positions').objectStore('positions').get([subject, articleId])
    request.onsuccess = () => {
      const value: unknown = request.result
      resolve(
        isRecord(value) && value.fingerprint === fingerprint && typeof value.fraction === 'number'
          ? Math.min(1, Math.max(0, value.fraction))
          : 0,
      )
    }
    request.onerror = () => reject(request.error)
  })
}
