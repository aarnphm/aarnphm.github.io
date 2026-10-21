import type { TrainingCalendarEnv } from '../training-calendar'

export type Env = {
  LFS_BUCKET_URL?: string
  KEEP_HEADERS?: string
  PUBLIC_BASE_URL?: string
  LEAN_VERIFY_ORIGIN?: string
  LEAN_VERIFY_TOKEN?: string
  MCP_RATE_LIMITER: RateLimit
} & Cloudflare.Env &
  TrainingCalendarEnv
