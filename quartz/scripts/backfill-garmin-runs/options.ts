import { type Sport } from '../../plugins/stores/strava'
import { requireStravaId } from '../../util/strava-id'

const DEFAULT_UPLOAD_DELAY_MS = 1500

export interface Args {
  write: boolean
  sport: BackfillSport
  since: string | null
  limit: number
  ids: Set<string>
  outputDir: string | null
  uploadDelayMs: number
}

export type BackfillSport = Extract<Sport, 'run' | 'swim'>

function usage(): string {
  return [
    'usage: pnpm garmin:backfill-runs -- [--write] [--sport run|swim] [--since YYYY-MM-DD] [--limit N] [--id ID]',
    '',
    'defaults to dry-run. --write uploads generated TCX files to Garmin Connect.',
  ].join('\n')
}

export function parseArgs(argv: string[]): Args {
  const args: Args = {
    write: false,
    sport: 'run',
    since: null,
    limit: 0,
    ids: new Set(),
    outputDir: null,
    uploadDelayMs: envNumber('GARMIN_BACKFILL_UPLOAD_DELAY_MS', DEFAULT_UPLOAD_DELAY_MS),
  }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--') continue
    if (arg === '--write') args.write = true
    else if (arg === '--dry-run') args.write = false
    else if (arg === '--sport') args.sport = parseSport(readArgValue(argv, ++i, arg))
    else if (arg === '--since') args.since = readArgValue(argv, ++i, arg)
    else if (arg === '--limit') args.limit = positiveInteger(readArgValue(argv, ++i, arg), arg)
    else if (arg === '--id') args.ids.add(requireStravaId(readArgValue(argv, ++i, arg)))
    else if (arg === '--ids') {
      for (const id of readArgValue(argv, ++i, arg).split(','))
        if (id.trim()) args.ids.add(requireStravaId(id.trim()))
    } else if (arg === '--out') args.outputDir = readArgValue(argv, ++i, arg)
    else if (arg === '--delay-ms')
      args.uploadDelayMs = nonnegativeInteger(readArgValue(argv, ++i, arg), arg)
    else if (arg === '--help' || arg === '-h') {
      console.log(usage())
      process.exit(0)
    } else throw new Error(`unknown argument ${arg}\n${usage()}`)
  }
  if (args.since && !/^\d{4}-\d{2}-\d{2}$/.test(args.since))
    throw new Error(`--since must be YYYY-MM-DD, got ${args.since}`)
  return args
}

function parseSport(value: string): BackfillSport {
  if (value === 'run' || value === 'swim') return value
  throw new Error(`--sport must be run or swim, got ${value}`)
}

function readArgValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name]
  if (!value?.trim()) return fallback
  return nonnegativeInteger(value, name)
}

function nonnegativeInteger(value: string, name: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0)
    throw new Error(`${name} must be a nonnegative integer`)
  return parsed
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0)
    throw new Error(`${name} must be a positive integer`)
  return parsed
}
