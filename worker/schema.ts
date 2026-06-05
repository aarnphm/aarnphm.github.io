import { sqliteTable, text, integer, real, index, primaryKey } from 'drizzle-orm/sqlite-core'

export const githubUsers = sqliteTable(
  'github_users',
  {
    login: text('login').primaryKey(),
    displayName: text('display_name'),
    avatarUrl: text('avatar_url'),
    lastSeenAt: integer('last_seen_at').notNull(),
    firstSeenAt: integer('first_seen_at').notNull(),
  },
  table => [index('idx_github_users_last_seen').on(table.lastSeenAt)],
)

export const comments = sqliteTable(
  'comments',
  {
    id: text('id').primaryKey(),
    pageId: text('pageId').notNull(),
    parentId: text('parentId'),
    anchorHash: text('anchorHash').notNull(),
    anchorStart: integer('anchorStart').notNull(),
    anchorEnd: integer('anchorEnd').notNull(),
    anchorText: text('anchorText').notNull(),
    content: text('content').notNull(),
    author: text('author').notNull(),
    createdAt: integer('createdAt').notNull(),
    updatedAt: integer('updatedAt'),
    deletedAt: integer('deletedAt'),
    resolvedAt: integer('resolvedAt'),
    anchor: text('anchor'),
    orphaned: integer('orphaned', { mode: 'boolean' }),
    lastRecoveredAt: integer('lastRecoveredAt'),
  },
  table => [
    index('idx_comments_page').on(table.pageId, table.createdAt),
    index('idx_comments_parent').on(table.parentId),
    index('idx_comments_hash').on(table.anchorHash),
  ],
)

export const flashcardReviews = sqliteTable(
  'flashcard_reviews',
  {
    login: text('login').notNull(),
    cardId: text('card_id').notNull(),
    deckSlug: text('deck_slug').notNull(),
    stability: real('stability').notNull(),
    difficulty: real('difficulty').notNull(),
    due: integer('due').notNull(),
    state: integer('state').notNull(),
    reps: integer('reps').notNull(),
    lapses: integer('lapses').notNull(),
    learningSteps: integer('learning_steps').notNull(),
    lastReviewedAt: integer('last_reviewed_at').notNull(),
  },
  table => [
    primaryKey({ columns: [table.login, table.cardId] }),
    index('idx_fc_due').on(table.login, table.due),
  ],
)
