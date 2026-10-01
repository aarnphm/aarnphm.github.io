import {
  newPdfMarkId,
  type PdfMark,
  type PdfMarkInput,
  type PdfMarkKind,
  type PdfMarksResponse,
  type PdfMarkTarget,
  type PdfMarkVisibility,
} from '../../util/pdf-marks'
import { conflictMark, deleteMark, MarkApiError, putMark } from './marks'
import { loadPending, removePending, savePending } from './store'

export type SaveStatus = 'saved' | 'saving' | 'pending' | 'conflict' | 'error'

export interface MarkEntry {
  mark: PdfMark
  /** Last revision the server acknowledged; 0 until the first save lands. */
  revision: number
  status: SaveStatus
  error?: string
  /** The server's copy after a 409, or null when another device deleted it. */
  conflict?: PdfMark | null
}

interface Deletion {
  entry: MarkEntry
  timer: number
}

const SAVE_DELAY_MS = 600
const UNDO_WINDOW_MS = 6000

function sameContent(mark: PdfMark, input: PdfMarkInput): boolean {
  return (
    mark.body === input.body &&
    mark.kind === input.kind &&
    mark.visibility === input.visibility &&
    JSON.stringify(mark.target) === JSON.stringify(input.target)
  )
}

export const saveStatusLabel: Record<SaveStatus, string> = {
  saved: 'saved',
  saving: 'saving…',
  pending: 'offline, will sync',
  conflict: 'changed elsewhere',
  error: 'not saved',
}

/** Client copy of one document's marks: optimistic edits, serialized saves, an offline queue. */
export class MarkBook {
  readonly entries = new Map<string, MarkEntry>()
  canWrite = false
  signInUrl = '/comments/github/login'
  version = 0
  private readonly listeners = new Set<() => void>()
  private readonly timers = new Map<string, number>()
  private readonly inflight = new Map<string, Promise<void>>()
  private readonly dirty = new Set<string>()
  private readonly deletions: Deletion[] = []

  constructor(
    readonly src: string,
    readonly doc: string,
  ) {}

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private emit() {
    this.version += 1
    for (const listener of this.listeners) listener()
  }

  async load(response: PdfMarksResponse) {
    this.canWrite = response.canWrite
    this.signInUrl = response.signInUrl
    for (const mark of response.marks) {
      this.entries.set(mark.id, { mark, revision: mark.revision, status: 'saved' })
    }
    const pending = this.canWrite ? await loadPending(this.src).catch(() => []) : []
    for (const op of pending) {
      if (op.type === 'delete') {
        this.entries.delete(op.id)
        void this.sendDelete(op.id, op.revision)
        continue
      }
      if (op.input.doc !== this.doc) continue
      const now = op.queuedAt || Date.now()
      const known = this.entries.get(op.id)
      this.entries.set(op.id, {
        mark: {
          id: op.id,
          doc: op.input.doc,
          src: op.input.src,
          kind: op.input.kind,
          target: op.input.target,
          body: op.input.body,
          visibility: op.input.visibility,
          revision: op.input.revision,
          createdAt: known?.mark.createdAt ?? now,
          updatedAt: now,
        },
        revision: op.input.revision,
        status: 'pending',
      })
    }
    this.emit()
    if (pending.length > 0) this.retry()
  }

  /** Sorted in reading order: page, then the first box from the top, then creation. */
  sorted(order: (mark: PdfMark) => number): MarkEntry[] {
    return [...this.entries.values()].sort(
      (a, b) =>
        a.mark.target.page - b.mark.target.page ||
        order(a.mark) - order(b.mark) ||
        a.mark.createdAt - b.mark.createdAt,
    )
  }

  create(
    kind: PdfMarkKind,
    target: PdfMarkTarget,
    visibility: PdfMarkVisibility = 'public',
    id = newPdfMarkId(),
    revision = 0,
  ): MarkEntry {
    const now = Date.now()
    const entry: MarkEntry = {
      mark: {
        id,
        doc: this.doc,
        src: this.src,
        kind,
        target,
        body: '',
        visibility,
        revision,
        createdAt: now,
        updatedAt: now,
      },
      revision,
      status: 'saving',
    }
    this.entries.set(id, entry)
    this.emit()
    void this.save(entry)
    return entry
  }

  /** Carries a mark from an older copy of the PDF onto this one, keeping its id and revision. */
  adopt(mark: PdfMark, target: PdfMarkTarget): MarkEntry {
    const entry: MarkEntry = {
      mark: { ...mark, doc: this.doc, src: this.src, target },
      revision: mark.revision,
      status: 'saving',
    }
    this.entries.set(mark.id, entry)
    this.emit()
    void this.save(entry)
    return entry
  }

  update(
    id: string,
    patch: Partial<Pick<PdfMark, 'body' | 'kind' | 'visibility' | 'target'>>,
    immediate = false,
  ) {
    const entry = this.entries.get(id)
    if (!entry || entry.status === 'conflict') return
    entry.mark = { ...entry.mark, ...patch, updatedAt: Date.now() }
    this.emit()
    window.clearTimeout(this.timers.get(id))
    if (immediate) {
      this.timers.delete(id)
      void this.save(entry)
    } else {
      this.timers.set(
        id,
        window.setTimeout(() => {
          this.timers.delete(id)
          void this.save(entry)
        }, SAVE_DELAY_MS),
      )
    }
  }

  private input(entry: MarkEntry): PdfMarkInput {
    const { kind, target, body, visibility } = entry.mark
    return {
      src: this.src,
      doc: this.doc,
      revision: entry.revision,
      kind,
      target,
      body,
      visibility,
    }
  }

  private async save(entry: MarkEntry): Promise<void> {
    const id = entry.mark.id
    if (this.inflight.has(id)) {
      this.dirty.add(id)
      return
    }
    const input = this.input(entry)
    entry.status = 'saving'
    entry.error = undefined
    this.emit()
    const run = (async () => {
      // Queue first, so a tab closed mid-request still has the edit on next load.
      await savePending({ id, src: this.src, type: 'put', input, queuedAt: Date.now() }).catch(
        () => undefined,
      )
      try {
        const saved = await putMark(id, input)
        entry.revision = saved.revision
        // Edits typed while the request was in flight stay local; their own save is queued.
        if (sameContent(entry.mark, input)) {
          entry.mark = saved
          entry.status = 'saved'
        }
        if (!this.timers.has(id) && !this.dirty.has(id)) {
          await removePending(id).catch(() => undefined)
        }
      } catch (error) {
        if (!(error instanceof MarkApiError)) throw error
        if (error.retryable) {
          entry.status = 'pending'
        } else {
          await removePending(id).catch(() => undefined)
          if (error.status === 409 && error.code === 'conflict') {
            entry.status = 'conflict'
            entry.conflict = conflictMark(error)
          } else if (error.status === 401) {
            this.canWrite = false
            entry.status = 'error'
            entry.error = 'Signed out. Sign in again to save.'
          } else {
            entry.status = 'error'
            entry.error = error.message
          }
        }
      }
    })()
    this.inflight.set(id, run)
    try {
      await run
    } finally {
      this.inflight.delete(id)
      this.emit()
      // `run` reassigns the status after the narrowing above, so widen it again.
      const status = entry.status as SaveStatus
      if (this.dirty.delete(id) && this.entries.get(id) === entry && status !== 'conflict') {
        void this.save(entry)
      }
    }
  }

  resolve(id: string, choice: 'mine' | 'theirs') {
    const entry = this.entries.get(id)
    if (!entry || entry.status !== 'conflict') return
    const theirs = entry.conflict
    entry.conflict = undefined
    if (choice === 'theirs') {
      if (theirs) {
        this.entries.set(id, { mark: theirs, revision: theirs.revision, status: 'saved' })
      } else {
        this.entries.delete(id)
      }
      this.emit()
      return
    }
    if (theirs) {
      entry.revision = theirs.revision
      void this.save(entry)
      return
    }
    // The server copy is gone; keep the words under a fresh id.
    this.entries.delete(id)
    const copy = this.create(entry.mark.kind, entry.mark.target, entry.mark.visibility)
    this.update(copy.mark.id, { body: entry.mark.body }, true)
  }

  /** Hides the mark now and deletes it after the undo window. */
  remove(id: string) {
    const entry = this.entries.get(id)
    if (!entry) return
    window.clearTimeout(this.timers.get(id))
    this.timers.delete(id)
    this.entries.delete(id)
    const deletion: Deletion = {
      entry,
      timer: window.setTimeout(() => void this.commit(deletion), UNDO_WINDOW_MS),
    }
    this.deletions.push(deletion)
    this.emit()
  }

  get canUndo(): boolean {
    return this.deletions.length > 0
  }

  undo(): MarkEntry | null {
    const deletion = this.deletions.pop()
    if (!deletion) return null
    window.clearTimeout(deletion.timer)
    this.entries.set(deletion.entry.mark.id, deletion.entry)
    this.emit()
    return deletion.entry
  }

  private async commit(deletion: Deletion) {
    const at = this.deletions.indexOf(deletion)
    if (at < 0) return
    this.deletions.splice(at, 1)
    const { entry } = deletion
    await this.inflight.get(entry.mark.id)?.catch(() => undefined)
    if (entry.revision === 0) {
      await removePending(entry.mark.id).catch(() => undefined)
      return
    }
    await this.sendDelete(entry.mark.id, entry.revision)
  }

  private async sendDelete(id: string, revision: number) {
    await savePending({ id, src: this.src, type: 'delete', revision, queuedAt: Date.now() }).catch(
      () => undefined,
    )
    try {
      await deleteMark(id, revision)
      await removePending(id).catch(() => undefined)
    } catch (error) {
      if (error instanceof MarkApiError && error.retryable) return
      await removePending(id).catch(() => undefined)
      if (error instanceof MarkApiError && error.status === 409 && !error.body.deleted) {
        const current = conflictMark(error)
        if (current) {
          this.entries.set(id, {
            mark: current,
            revision: current.revision,
            status: 'error',
            error: 'Changed on another device, so it was kept.',
          })
          this.emit()
        }
      }
    }
  }

  /** Sends deletions that are still inside the undo window; navigation away commits them. */
  flush() {
    // `commit` splices `deletions` as it starts, so walk a copy.
    for (const deletion of this.deletions.slice()) {
      window.clearTimeout(deletion.timer)
      void this.commit(deletion)
    }
    for (const [id, timer] of this.timers) {
      window.clearTimeout(timer)
      const entry = this.entries.get(id)
      if (entry) void this.save(entry)
    }
    this.timers.clear()
  }

  retry() {
    for (const entry of this.entries.values()) {
      if (entry.status === 'pending') void this.save(entry)
    }
    void loadPending(this.src)
      .then(ops => {
        for (const op of ops) if (op.type === 'delete') void this.sendDelete(op.id, op.revision)
      })
      .catch(() => undefined)
  }
}
