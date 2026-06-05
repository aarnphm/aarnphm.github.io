import { and, eq } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/d1'
import { Card, CardInput, Grade, createEmptyCard, fsrs, generatorParameters } from 'ts-fsrs'
import { flashcardReviews } from './schema'

const scheduler = fsrs(generatorParameters({ enable_fuzz: true }))

type ReviewRow = typeof flashcardReviews.$inferSelect

function normalizeLogin(raw: unknown): string | null {
  if (typeof raw !== 'string') return null
  const trimmed = raw.trim()
  if (!trimmed || trimmed.length > 128) return null
  return trimmed
}

function publicState(row: ReviewRow) {
  return {
    cardId: row.cardId,
    due: row.due,
    state: row.state,
    reps: row.reps,
    lapses: row.lapses,
    lastReviewedAt: row.lastReviewedAt,
  }
}

export async function handleFlashcardsState(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url)
  const deck = url.searchParams.get('deck')
  const login = normalizeLogin(url.searchParams.get('login'))
  if (!deck) return new Response('deck required', { status: 400 })
  if (!login) return Response.json({ states: [] })

  const db = drizzle(env.COMMENTS_ROOM)
  const rows = await db
    .select()
    .from(flashcardReviews)
    .where(and(eq(flashcardReviews.login, login), eq(flashcardReviews.deckSlug, deck)))
  return Response.json({ states: rows.map(publicState) })
}

export async function handleFlashcardsReview(request: Request, env: Env): Promise<Response> {
  if (request.method !== 'POST') return new Response('method not allowed', { status: 405 })

  let body: { cardId?: unknown; deckSlug?: unknown; grade?: unknown; login?: unknown }
  try {
    body = (await request.json()) as typeof body
  } catch {
    return new Response('invalid json', { status: 400 })
  }

  const login = normalizeLogin(body.login)
  const cardId = typeof body.cardId === 'string' ? body.cardId : null
  const deckSlug = typeof body.deckSlug === 'string' ? body.deckSlug : null
  const grade = typeof body.grade === 'number' ? body.grade : NaN
  if (!login) return new Response('authentication required', { status: 401 })
  if (!cardId || !deckSlug) return new Response('cardId and deckSlug required', { status: 400 })
  if (!Number.isInteger(grade) || grade < 1 || grade > 4) {
    return new Response('invalid grade', { status: 400 })
  }

  const db = drizzle(env.COMMENTS_ROOM)
  const now = new Date()
  const prior = await db
    .select()
    .from(flashcardReviews)
    .where(and(eq(flashcardReviews.login, login), eq(flashcardReviews.cardId, cardId)))
    .get()

  const source: Card | CardInput = prior
    ? {
        due: prior.due,
        stability: prior.stability,
        difficulty: prior.difficulty,
        elapsed_days: 0,
        scheduled_days: 0,
        learning_steps: prior.learningSteps,
        reps: prior.reps,
        lapses: prior.lapses,
        state: prior.state,
        last_review: prior.lastReviewedAt,
      }
    : createEmptyCard(now)

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

  await db
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
    })

  return Response.json({ state: publicState(next) })
}
