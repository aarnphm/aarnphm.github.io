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
    index('idx_fc_deck').on(table.login, table.deckSlug),
  ],
)

// One row per graded review with the card's prior state, so an undo can restore it and
// measured retention has data to read. Prior columns are null when the card was new.
export const flashcardReviewLog = sqliteTable(
  'flashcard_review_log',
  {
    id: integer('id').primaryKey({ autoIncrement: true }),
    login: text('login').notNull(),
    cardId: text('card_id').notNull(),
    deckSlug: text('deck_slug').notNull(),
    grade: integer('grade').notNull(),
    reviewedAt: integer('reviewed_at').notNull(),
    priorState: integer('prior_state'),
    priorStability: real('prior_stability'),
    priorDifficulty: real('prior_difficulty'),
    priorDue: integer('prior_due'),
    priorReps: integer('prior_reps'),
    priorLapses: integer('prior_lapses'),
    priorLearningSteps: integer('prior_learning_steps'),
    priorReviewedAt: integer('prior_reviewed_at'),
  },
  table => [index('idx_fc_log_card').on(table.login, table.cardId, table.reviewedAt)],
)
