import type { Argv } from './ctx'

/**
 * The dev-only study endpoint (`quartz/scripts/course-study-server.ts`, started by dev.ts) listens
 * on the dev port plus this offset, beside wrangler at +707. Nothing on the deployed site writes
 * to the repo: built without `--watch` or `--serve`, pages carry no endpoint and dates are text.
 */
export const COURSE_STUDY_PORT_OFFSET = 709

export function courseStudyEndpoint(argv: Pick<Argv, 'watch' | 'serve' | 'port'>) {
  if (!argv.watch && !argv.serve) return undefined
  return `http://localhost:${argv.port + COURSE_STUDY_PORT_OFFSET}`
}

export const STUDY_DATE = /^\d{4}-\d{2}-\d{2}$/
export const COURSE_DIR = /^[a-z0-9][a-z0-9.-]*$/i

/** `POST /study`: set or clear one date in `lectures/index.md`. */
export interface StudyRequest {
  dir: string
  key: 'read' | 'problems'
  n: number
  date: string | null
}

/** `POST /promote`: write a new `lectures/<nn>.md` and its `.fc` drafts; never overwrites. */
export interface PromoteRequest {
  dir: string
  n: number
  md: string
  fc: string
}

export type ProblemState = 'open' | 'stuck' | 'solved'

/** One problem in `psets/<nn>.md` frontmatter. */
export interface ProblemRecord {
  state: ProblemState
  date?: string
  /** Day the official solution was opened for this problem. */
  reveal?: string
}

/** `POST /pset`: create `psets/<nn>.md` when missing, then set one problem's state or reveal. */
export interface PsetRequest {
  dir: string
  n: number
  problem?: number
  state?: ProblemState
  reveal?: string
  date?: string
}

export function parseProblems(frontmatter: unknown): Map<number, ProblemRecord> {
  const out = new Map<number, ProblemRecord>()
  const problems =
    typeof frontmatter === 'object' && frontmatter !== null
      ? (frontmatter as Record<string, unknown>).problems
      : undefined
  if (typeof problems !== 'object' || problems === null) return out
  for (const [raw, value] of Object.entries(problems)) {
    const n = Number(raw)
    if (!Number.isInteger(n) || n < 1 || typeof value !== 'object' || value === null) continue
    const record = value as Record<string, unknown>
    const state = record.state
    if (state !== 'open' && state !== 'stuck' && state !== 'solved') continue
    const day = (v: unknown) =>
      v instanceof Date
        ? v.toISOString().slice(0, 10)
        : typeof v === 'string' && STUDY_DATE.test(v.slice(0, 10))
          ? v.slice(0, 10)
          : undefined
    const date = day(record.date)
    const reveal = day(record.reveal)
    out.set(n, { state, ...(date ? { date } : {}), ...(reveal ? { reveal } : {}) })
  }
  return out
}
