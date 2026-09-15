import { and, asc, desc, eq, isNull, lt, or, sql } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/d1'
import type {
  ArenaNote,
  ArenaNoteOccurrence,
  ArenaNoteQuote,
  ArenaReadLink,
} from '../quartz/util/arena-reader'
import { arenaNotes, arenaReadLinks } from './schema/arena-reader'

export interface ArenaReadLinkInput {
  read: boolean
  revision: number
}

export interface ArenaNoteInput {
  articleId: string
  sourceUrl: string
  body: string
  snapshotId: string | null
  quote: ArenaNoteQuote | null
  occurrence: ArenaNoteOccurrence | null
  revision: number
  ready: boolean
}

export interface ArenaNoteExportBundle {
  schemaVersion: 1
  generatedAt: number
  notes: (ArenaNote & { bodyHash: string })[]
}

export interface ArenaNoteExportAcknowledgement {
  revision: number
  receipt: string
}

export class ArenaReaderConflictError extends Error {
  readonly current: ArenaReadLink | ArenaNote | null

  constructor(current: ArenaReadLink | ArenaNote | null) {
    super('This record changed since the supplied revision.')
    this.name = 'ArenaReaderConflictError'
    this.current = current
  }
}

export class ArenaReaderNoteNotFoundError extends Error {
  constructor() {
    super('This note does not exist.')
    this.name = 'ArenaReaderNoteNotFoundError'
  }
}

type ReadRow = typeof arenaReadLinks.$inferSelect
type NoteRow = typeof arenaNotes.$inferSelect

function readLink({ subject: _subject, ...row }: ReadRow): ArenaReadLink {
  return row
}

function note({ subject: _subject, ...row }: NoteRow): ArenaNote {
  return row
}

async function findReadLink(
  database: D1Database,
  subject: string,
  articleId: string,
): Promise<ArenaReadLink | null> {
  const row = await drizzle(database)
    .select()
    .from(arenaReadLinks)
    .where(and(eq(arenaReadLinks.subject, subject), eq(arenaReadLinks.articleId, articleId)))
    .get()
  return row ? readLink(row) : null
}

async function findNote(
  database: D1Database,
  subject: string,
  id: string,
): Promise<ArenaNote | null> {
  const row = await drizzle(database)
    .select()
    .from(arenaNotes)
    .where(and(eq(arenaNotes.subject, subject), eq(arenaNotes.id, id)))
    .get()
  return row ? note(row) : null
}

export async function listReadLinks(
  database: D1Database,
  subject: string,
): Promise<ArenaReadLink[]> {
  const rows = await drizzle(database)
    .select()
    .from(arenaReadLinks)
    .where(eq(arenaReadLinks.subject, subject))
    .orderBy(asc(arenaReadLinks.articleId))
  return rows.map(readLink)
}

export async function setReadLink(
  database: D1Database,
  subject: string,
  articleId: string,
  input: ArenaReadLinkInput,
): Promise<ArenaReadLink> {
  const db = drizzle(database)
  const now = Date.now()
  const revision = input.revision + 1
  const rows =
    input.revision === 0
      ? await db
          .insert(arenaReadLinks)
          .values({ subject, articleId, readAt: input.read ? now : null, updatedAt: now, revision })
          .onConflictDoNothing()
          .returning()
      : await db
          .update(arenaReadLinks)
          .set({
            readAt: input.read ? sql`coalesce(${arenaReadLinks.readAt}, ${now})` : null,
            updatedAt: now,
            revision,
          })
          .where(
            and(
              eq(arenaReadLinks.subject, subject),
              eq(arenaReadLinks.articleId, articleId),
              eq(arenaReadLinks.revision, input.revision),
            ),
          )
          .returning()

  if (rows[0]) return readLink(rows[0])
  const current = await findReadLink(database, subject, articleId)
  if (current?.revision === revision && (current.readAt !== null) === input.read) return current
  throw new ArenaReaderConflictError(current)
}

export async function listArenaNotes(
  database: D1Database,
  subject: string,
  articleId?: string,
): Promise<ArenaNote[]> {
  const rows = await drizzle(database)
    .select()
    .from(arenaNotes)
    .where(
      and(
        eq(arenaNotes.subject, subject),
        articleId === undefined ? undefined : eq(arenaNotes.articleId, articleId),
      ),
    )
    .orderBy(desc(arenaNotes.updatedAt), asc(arenaNotes.id))
  return rows.map(note)
}

function sameNoteContent(current: ArenaNote, input: ArenaNoteInput): boolean {
  return (
    current.deletedAt === null &&
    current.articleId === input.articleId &&
    current.sourceUrl === input.sourceUrl &&
    current.body === input.body &&
    current.snapshotId === input.snapshotId &&
    current.quote?.exact === input.quote?.exact &&
    current.quote?.prefix === input.quote?.prefix &&
    current.quote?.suffix === input.quote?.suffix &&
    current.occurrence?.channelSlug === input.occurrence?.channelSlug &&
    current.occurrence?.blockId === input.occurrence?.blockId &&
    (current.readyRevision === current.revision) === input.ready
  )
}

export async function saveArenaNote(
  database: D1Database,
  subject: string,
  id: string,
  input: ArenaNoteInput,
): Promise<ArenaNote> {
  const db = drizzle(database)
  const now = Date.now()
  const revision = input.revision + 1
  const next = {
    body: input.body,
    snapshotId: input.snapshotId,
    quote: input.quote,
    occurrence: input.occurrence,
    updatedAt: now,
    revision,
    readyRevision: input.ready ? revision : null,
  }
  const rows =
    input.revision === 0
      ? await db
          .insert(arenaNotes)
          .values({
            subject,
            id,
            articleId: input.articleId,
            sourceUrl: input.sourceUrl,
            createdAt: now,
            ...next,
          })
          .onConflictDoNothing()
          .returning()
      : await db
          .update(arenaNotes)
          .set(next)
          .where(
            and(
              eq(arenaNotes.subject, subject),
              eq(arenaNotes.id, id),
              eq(arenaNotes.revision, input.revision),
              eq(arenaNotes.articleId, input.articleId),
              eq(arenaNotes.sourceUrl, input.sourceUrl),
              isNull(arenaNotes.deletedAt),
            ),
          )
          .returning()

  if (rows[0]) return note(rows[0])
  const current = await findNote(database, subject, id)
  if (current?.revision === revision && sameNoteContent(current, input)) return current
  throw new ArenaReaderConflictError(current)
}

export async function deleteArenaNote(
  database: D1Database,
  subject: string,
  id: string,
  revision: number,
): Promise<ArenaNote> {
  const now = Date.now()
  const rows = await drizzle(database)
    .update(arenaNotes)
    .set({ deletedAt: now, updatedAt: now, revision: revision + 1, readyRevision: null })
    .where(
      and(
        eq(arenaNotes.subject, subject),
        eq(arenaNotes.id, id),
        eq(arenaNotes.revision, revision),
        isNull(arenaNotes.deletedAt),
      ),
    )
    .returning()

  if (rows[0]) return note(rows[0])
  const current = await findNote(database, subject, id)
  if (!current) throw new ArenaReaderNoteNotFoundError()
  if (current.revision === revision + 1 && current.deletedAt !== null) return current
  throw new ArenaReaderConflictError(current)
}

export async function exportReadyArenaNotes(
  database: D1Database,
  subject: string,
): Promise<ArenaNoteExportBundle> {
  const rows = await drizzle(database)
    .select()
    .from(arenaNotes)
    .where(
      and(
        eq(arenaNotes.subject, subject),
        isNull(arenaNotes.deletedAt),
        eq(arenaNotes.readyRevision, arenaNotes.revision),
        or(
          isNull(arenaNotes.exportedRevision),
          lt(arenaNotes.exportedRevision, arenaNotes.revision),
        ),
      ),
    )
    .orderBy(asc(arenaNotes.updatedAt), asc(arenaNotes.id))
  const notes = await Promise.all(
    rows.map(async row => {
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(row.body))
      const bodyHash = Array.from(new Uint8Array(digest), byte =>
        byte.toString(16).padStart(2, '0'),
      ).join('')
      return { ...note(row), bodyHash }
    }),
  )
  return { schemaVersion: 1, generatedAt: Date.now(), notes }
}

export async function acknowledgeArenaNoteExport(
  database: D1Database,
  subject: string,
  id: string,
  input: ArenaNoteExportAcknowledgement,
): Promise<ArenaNote> {
  const rows = await drizzle(database)
    .update(arenaNotes)
    .set({
      exportedRevision: input.revision,
      exportReceipt: input.receipt,
      readyRevision: sql`CASE WHEN ${arenaNotes.readyRevision} = ${input.revision} THEN NULL ELSE ${arenaNotes.readyRevision} END`,
    })
    .where(
      and(
        eq(arenaNotes.subject, subject),
        eq(arenaNotes.id, id),
        sql`${arenaNotes.revision} >= ${input.revision}`,
        or(isNull(arenaNotes.exportedRevision), lt(arenaNotes.exportedRevision, input.revision)),
        or(
          sql`${arenaNotes.revision} > ${input.revision}`,
          eq(arenaNotes.readyRevision, input.revision),
        ),
      ),
    )
    .returning()

  if (rows[0]) return note(rows[0])
  const current = await findNote(database, subject, id)
  if (!current) throw new ArenaReaderNoteNotFoundError()
  if (current.exportedRevision === input.revision && current.exportReceipt === input.receipt) {
    return current
  }
  throw new ArenaReaderConflictError(current)
}
