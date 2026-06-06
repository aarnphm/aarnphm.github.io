import { sqliteTable, text, integer, real, index, primaryKey } from 'drizzle-orm/sqlite-core'

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
