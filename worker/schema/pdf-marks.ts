import { sql } from 'drizzle-orm'
import { check, index, integer, sqliteTable, text } from 'drizzle-orm/sqlite-core'
import type { PdfMarkKind, PdfMarkTarget, PdfMarkVisibility } from '../../quartz/util/pdf-marks'

export const pdfMarks = sqliteTable(
  'pdf_marks',
  {
    id: text('mark_id').primaryKey(),
    owner: text('owner').notNull(),
    doc: text('doc').notNull(),
    src: text('src').notNull(),
    kind: text('kind').$type<PdfMarkKind>().notNull(),
    page: integer('page').notNull(),
    target: text('target', { mode: 'json' }).$type<PdfMarkTarget>().notNull(),
    body: text('body').notNull().default(''),
    visibility: text('visibility').$type<PdfMarkVisibility>().notNull(),
    revision: integer('revision').notNull(),
    createdAt: integer('created_at').notNull(),
    updatedAt: integer('updated_at').notNull(),
    deletedAt: integer('deleted_at'),
  },
  table => [
    index('idx_pdf_marks_doc').on(table.doc, table.deletedAt, table.visibility, table.page),
    index('idx_pdf_marks_src').on(table.owner, table.src, table.deletedAt),
    check('pdf_marks_kind', sql`${table.kind} IN ('mark', 'question', 'contra')`),
    check('pdf_marks_visibility', sql`${table.visibility} IN ('public', 'private')`),
    check('pdf_marks_revision', sql`${table.revision} > 0`),
  ],
)
