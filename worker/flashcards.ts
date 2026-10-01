import { and, eq } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/d1'
import { Card, CardInput, Grade, createEmptyCard, fsrs, generatorParameters } from 'ts-fsrs'
import { getOwnerSessionLogin, isArenaReaderMutationAllowed } from './arena-reader-auth'
import { flashcardReviews } from './schema'

const scheduler = fsrs(generatorParameters({ enable_fuzz: true }))

// Every visitor requests the same state URL; a shared cache would hand the owner's rows to anyone.
const privateHeaders = { 'Cache-Control': 'private, no-store' }

type ReviewRow = typeof flashcardReviews.$inferSelect

function textResponse(body: string, status: number): Response {
  return new Response(body, { status, headers: privateHeaders })
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
  const deck = new URL(request.url).searchParams.get('deck')
  if (!deck) return textResponse('deck required', 400)
  const login = await getOwnerSessionLogin(request, env)
  if (!login) return Response.json({ login: null, states: [] }, { headers: privateHeaders })

  const db = drizzle(env.FLASHCARDS)
  const rows = await db
    .select()
    .from(flashcardReviews)
    .where(and(eq(flashcardReviews.login, login), eq(flashcardReviews.deckSlug, deck)))
  return Response.json({ login, states: rows.map(publicState) }, { headers: privateHeaders })
}

export async function handleFlashcardsReview(request: Request, env: Env): Promise<Response> {
  if (request.method !== 'POST') return textResponse('method not allowed', 405)
  if (!isArenaReaderMutationAllowed(request))
    return textResponse('cross-origin review rejected', 403)
  const login = await getOwnerSessionLogin(request, env)
  if (!login) return textResponse('authentication required', 401)

  let body: { cardId?: unknown; deckSlug?: unknown; grade?: unknown }
  try {
    body = (await request.json()) as typeof body
  } catch {
    return textResponse('invalid json', 400)
  }

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

  return Response.json({ state: publicState(next) }, { headers: privateHeaders })
}
