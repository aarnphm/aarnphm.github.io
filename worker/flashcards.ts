import { and, eq, gte, lt } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/d1'
import { Card, CardInput, Grade, createEmptyCard, fsrs, generatorParameters } from 'ts-fsrs'
import { getOwnerSessionLogin, isArenaReaderMutationAllowed } from './arena-reader-auth'
import { flashcardReviewLog, flashcardReviews } from './schema'

const scheduler = fsrs(generatorParameters({ enable_fuzz: true }))

// Every visitor requests the same state URL; a shared cache would hand the owner's rows to anyone.
const privateHeaders = { 'Cache-Control': 'private, no-store' }

type ReviewRow = typeof flashcardReviews.$inferSelect
type LogRow = typeof flashcardReviewLog.$inferSelect

function textResponse(body: string, status: number): Response {
  return new Response(body, { status, headers: privateHeaders })
}

function cardFromRow(row: ReviewRow): Card {
  return {
    due: new Date(row.due),
    stability: row.stability,
    difficulty: row.difficulty,
    elapsed_days: 0,
    scheduled_days: 0,
    learning_steps: row.learningSteps,
    reps: row.reps,
    lapses: row.lapses,
    state: row.state,
    last_review: new Date(row.lastReviewedAt),
  }
}

// `r` is the scheduler's own forgetting-curve estimate, so a readout cannot drift from the
// parameters that produced the schedule.
function publicState(row: ReviewRow, now: Date) {
  const r = scheduler.get_retrievability(cardFromRow(row), now, false)
  return {
    cardId: row.cardId,
    deckSlug: row.deckSlug,
    due: row.due,
    state: row.state,
    reps: row.reps,
    lapses: row.lapses,
    stability: row.stability,
    lastReviewedAt: row.lastReviewedAt,
    r: Number.isFinite(r) ? Math.round(r * 1000) / 1000 : null,
  }
}

// The range form keeps the (login, deck_slug) index usable and needs no LIKE escaping.
function prefixRange(prefix: string): { lo: string; hi: string } | null {
  if (prefix.length === 0) return null
  const last = prefix.charCodeAt(prefix.length - 1)
  return { lo: prefix, hi: prefix.slice(0, -1) + String.fromCharCode(last + 1) }
}

export async function handleFlashcardsState(request: Request, env: Env): Promise<Response> {
  const params = new URL(request.url).searchParams
  const deck = params.get('deck')
  const prefix = params.get('prefix')
  if ((deck === null) === (prefix === null)) return textResponse('deck or prefix required', 400)
  const login = await getOwnerSessionLogin(request, env)
  if (!login) return Response.json({ login: null, states: [] }, { headers: privateHeaders })

  const db = drizzle(env.FLASHCARDS)
  const range = prefix === null ? null : prefixRange(prefix)
  const scope =
    deck !== null
      ? eq(flashcardReviews.deckSlug, deck)
      : range
        ? and(gte(flashcardReviews.deckSlug, range.lo), lt(flashcardReviews.deckSlug, range.hi))
        : undefined
  const rows = await db
    .select()
    .from(flashcardReviews)
    .where(and(eq(flashcardReviews.login, login), scope))
  const now = new Date()
  return Response.json(
    { login, states: rows.map(row => publicState(row, now)) },
    { headers: privateHeaders },
  )
}

async function ownerMutation(
  request: Request,
  env: Env,
): Promise<{ login: string } | { error: Response }> {
  if (request.method !== 'POST') return { error: textResponse('method not allowed', 405) }
  if (!isArenaReaderMutationAllowed(request))
    return { error: textResponse('cross-origin review rejected', 403) }
  const login = await getOwnerSessionLogin(request, env)
  if (!login) return { error: textResponse('authentication required', 401) }
  return { login }
}

async function jsonBody<T extends object>(request: Request): Promise<T | null> {
  try {
    const body: unknown = await request.json()
    return body !== null && typeof body === 'object' ? (body as T) : null
  } catch {
    return null
  }
}

export async function handleFlashcardsReview(request: Request, env: Env): Promise<Response> {
  const auth = await ownerMutation(request, env)
  if ('error' in auth) return auth.error
  const { login } = auth

  const body = await jsonBody<{ cardId?: unknown; deckSlug?: unknown; grade?: unknown }>(request)
  if (!body) return textResponse('invalid json', 400)

  const cardId = typeof body.cardId === 'string' ? body.cardId : null
  const deckSlug = typeof body.deckSlug === 'string' ? body.deckSlug : null
  const grade = typeof body.grade === 'number' ? body.grade : NaN
  if (!cardId || !deckSlug) return textResponse('cardId and deckSlug required', 400)
  if (!Number.isInteger(grade) || grade < 1 || grade > 4) return textResponse('invalid grade', 400)

  const db = drizzle(env.FLASHCARDS)
  const now = new Date()
  const prior = await db
    .select()
    .from(flashcardReviews)
    .where(and(eq(flashcardReviews.login, login), eq(flashcardReviews.cardId, cardId)))
    .get()

  const source: Card | CardInput = prior ? cardFromRow(prior) : createEmptyCard(now)
  const { card } = scheduler.next(source, now, grade as Grade)
  const next: ReviewRow = {
    login,
    cardId,
    deckSlug,
    stability: card.stability,
    difficulty: card.difficulty,
    due: card.due.getTime(),
    state: card.state,
    reps: card.reps,
    lapses: card.lapses,
    learningSteps: card.learning_steps,
    lastReviewedAt: now.getTime(),
  }
  const logged: typeof flashcardReviewLog.$inferInsert = {
    login,
    cardId,
    deckSlug,
    grade,
    reviewedAt: now.getTime(),
    priorState: prior?.state ?? null,
    priorStability: prior?.stability ?? null,
    priorDifficulty: prior?.difficulty ?? null,
    priorDue: prior?.due ?? null,
    priorReps: prior?.reps ?? null,
    priorLapses: prior?.lapses ?? null,
    priorLearningSteps: prior?.learningSteps ?? null,
    priorReviewedAt: prior?.lastReviewedAt ?? null,
  }

  const [inserted] = await db.batch([
    db.insert(flashcardReviewLog).values(logged).returning({ id: flashcardReviewLog.id }),
    db
      .insert(flashcardReviews)
      .values(next)
      .onConflictDoUpdate({
        target: [flashcardReviews.login, flashcardReviews.cardId],
        set: {
          deckSlug: next.deckSlug,
          stability: next.stability,
          difficulty: next.difficulty,
          due: next.due,
          state: next.state,
          reps: next.reps,
          lapses: next.lapses,
          learningSteps: next.learningSteps,
          lastReviewedAt: next.lastReviewedAt,
        },
      }),
  ])

  return Response.json(
    { state: publicState(next, now), reviewId: inserted[0]?.id ?? null },
    { headers: privateHeaders },
  )
}

function restoredRow(log: LogRow): ReviewRow | null {
  if (log.priorState === null) return null
  return {
    login: log.login,
    cardId: log.cardId,
    deckSlug: log.deckSlug,
    stability: log.priorStability ?? 0,
    difficulty: log.priorDifficulty ?? 0,
    due: log.priorDue ?? log.reviewedAt,
    state: log.priorState,
    reps: log.priorReps ?? 0,
    lapses: log.priorLapses ?? 0,
    learningSteps: log.priorLearningSteps ?? 0,
    lastReviewedAt: log.priorReviewedAt ?? log.reviewedAt,
  }
}

// Only the newest review of a card can be undone: an older log row's prior state would
// discard every grade recorded after it.
export async function handleFlashcardsUndo(request: Request, env: Env): Promise<Response> {
  const auth = await ownerMutation(request, env)
  if ('error' in auth) return auth.error
  const { login } = auth

  const body = await jsonBody<{ reviewId?: unknown }>(request)
  if (!body) return textResponse('invalid json', 400)
  const reviewId = typeof body.reviewId === 'number' ? body.reviewId : NaN
  if (!Number.isInteger(reviewId)) return textResponse('reviewId required', 400)

  const db = drizzle(env.FLASHCARDS)
  const log = await db
    .select()
    .from(flashcardReviewLog)
    .where(and(eq(flashcardReviewLog.login, login), eq(flashcardReviewLog.id, reviewId)))
    .get()
  if (!log) return textResponse('review not found', 404)
  const newer = await db
    .select({ id: flashcardReviewLog.id })
    .from(flashcardReviewLog)
    .where(
      and(
        eq(flashcardReviewLog.login, login),
        eq(flashcardReviewLog.cardId, log.cardId),
        gte(flashcardReviewLog.reviewedAt, log.reviewedAt),
      ),
    )
  if (newer.some(row => row.id > log.id)) return textResponse('a later review exists', 409)

  const restored = restoredRow(log)
  const where = and(eq(flashcardReviews.login, login), eq(flashcardReviews.cardId, log.cardId))
  const dropLog = db.delete(flashcardReviewLog).where(eq(flashcardReviewLog.id, log.id))
  if (restored) {
    await db.batch([db.update(flashcardReviews).set(restored).where(where), dropLog])
  } else {
    await db.batch([db.delete(flashcardReviews).where(where), dropLog])
  }
  return Response.json(
    { state: restored ? publicState(restored, new Date()) : null },
    { headers: privateHeaders },
  )
}
