import { sql } from 'drizzle-orm'
import { check, index, integer, primaryKey, sqliteTable, text } from 'drizzle-orm/sqlite-core'
import type { ArenaNoteOccurrence, ArenaNoteQuote } from '../../quartz/util/arena-reader'

export const arenaReadLinks = sqliteTable(
  'arena_read_links',
  {
    subject: text('subject').notNull(),
    articleId: text('article_id').notNull(),
    readAt: integer('read_at'),
    updatedAt: integer('updated_at').notNull(),
    revision: integer('revision').notNull(),
  },
  table => [
    primaryKey({ columns: [table.subject, table.articleId] }),
    check('arena_read_links_revision', sql`${table.revision} > 0`),
  ],
)

export const arenaNotes = sqliteTable(
  'arena_notes',
  {
    subject: text('subject').notNull(),
    id: text('note_id').notNull(),
    articleId: text('article_id').notNull(),
    sourceUrl: text('source_url').notNull(),
    body: text('body').notNull(),
    snapshotId: text('snapshot_id'),
    quote: text('quote', { mode: 'json' }).$type<ArenaNoteQuote>(),
    occurrence: text('occurrence', { mode: 'json' }).$type<ArenaNoteOccurrence>(),
    createdAt: integer('created_at').notNull(),
    updatedAt: integer('updated_at').notNull(),
    revision: integer('revision').notNull(),
    readyRevision: integer('ready_revision'),
    exportedRevision: integer('exported_revision'),
    exportReceipt: text('export_receipt'),
    deletedAt: integer('deleted_at'),
  },
  table => [
    primaryKey({ columns: [table.subject, table.id] }),
    index('idx_arena_notes_article').on(table.subject, table.articleId, table.createdAt),
    index('idx_arena_notes_updated').on(table.subject, table.updatedAt),
    check('arena_notes_revision', sql`${table.revision} > 0`),
    check(
      'arena_notes_ready_revision',
      sql`${table.readyRevision} IS NULL OR ${table.readyRevision} = ${table.revision}`,
    ),
    check(
      'arena_notes_exported_revision',
      sql`${table.exportedRevision} IS NULL OR (${table.exportedRevision} > 0 AND ${table.exportedRevision} <= ${table.revision})`,
    ),
  ],
)
