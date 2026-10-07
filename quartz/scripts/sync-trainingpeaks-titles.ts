import { fromMarkdown } from 'mdast-util-from-markdown'
import fs from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { visit } from 'unist-util-visit'
import { parseTrackingBlock, type ManualSaunaEntry } from '../plugins/stores/tracking'
import { readStravaCacheFile } from '../util/strava-cache-file'
import { TrainingPeaksApi } from '../util/trainingpeaks-api'
import {
  selectTrainingPeaksTitleSources,
  trainingPeaksTitleCandidates,
  trainingPeaksSaunaDescription,
  trainingPeaksWorkoutSummary,
  trainingPeaksStrengthWorkoutSummary,
  trainingPeaksSupportedTitle,
  trainingPeaksTitleProtection,
  trainingPeaksStrengthTitleProtection,
  type TrainingPeaksTitleSource,
} from '../util/trainingpeaks-title-sync'

const ROOT = resolve(import.meta.dirname, '../..')
const CACHE = resolve(ROOT, 'quartz/.quartz-cache')

interface Args {
  write: boolean
  sourcesOnly: boolean
  since: string | null
  until: string | null
  ids: Set<number>
  limit: number | null
}

export function parseTrainingPeaksTitleArgs(argv: readonly string[]): Args {
  const args: Args = {
    write: false,
    sourcesOnly: false,
    since: null,
    until: null,
    ids: new Set(),
    limit: null,
  }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--') continue
    if (arg === '--write') args.write = true
    else if (arg === '--dry-run') args.write = false
    else if (arg === '--sources') args.sourcesOnly = true
    else if (['--since', '--until', '--id', '--limit'].includes(arg)) {
      const value = argv[++i]
      if (!value || value.startsWith('--')) throw new Error(`${arg} requires a value`)
      if (arg === '--since' || arg === '--until') {
        if (
          !/^\d{4}-\d{2}-\d{2}$/.test(value) ||
          !Number.isFinite(Date.parse(value)) ||
          new Date(value).toISOString().slice(0, 10) !== value
        )
          throw new Error(`${arg} requires a valid YYYY-MM-DD date`)
        if (arg === '--since') args.since = value
        else args.until = value
      } else {
        const number = Number(value)
        if (!Number.isSafeInteger(number) || number <= 0)
          throw new Error(`${arg} requires a positive integer`)
        if (arg === '--id') args.ids.add(number)
        else args.limit = number
      }
    } else throw new Error(`Unknown TrainingPeaks title argument: ${arg}`)
  }
  if (args.since && args.until && args.since > args.until)
    throw new Error('--since must precede --until')
  if (args.sourcesOnly && args.write) throw new Error('--sources cannot be combined with --write')
  return args
}

export function trainingPeaksSaunaTracking(markdown: string): ManualSaunaEntry[] {
  const entries: ManualSaunaEntry[] = []
  visit(fromMarkdown(markdown), 'code', node => {
    if (node.lang !== 'tracking') return
    const sauna = parseTrackingBlock(node.meta, node.value)?.sauna
    if (sauna) entries.push(sauna)
  })
  return entries
}

interface TitleResult {
  stravaId: number
  date: string
  title: string
  workoutId?: string
  status:
    | 'updated'
    | 'unchanged'
    | 'planned'
    | 'protected'
    | 'unmatched'
    | 'ambiguous'
    | 'skipped'
    | 'failed'
  reason?: string
  before?: { title: string; description: string | null }
  after?: { title: string; description: string | null }
  candidateWorkoutIds?: string[]
  titleAdjusted?: boolean
}

interface TitleRun {
  athleteId?: number
  results: TitleResult[]
  error?: string
}

async function syncTitles(
  api: TrainingPeaksApi,
  sources: readonly TrainingPeaksTitleSource[],
  allSources: readonly TrainingPeaksTitleSource[],
  write: boolean,
  state: TitleRun,
): Promise<void> {
  const athleteId = await api.athleteId()
  state.athleteId = athleteId
  const dates = sources.map(source => source.date).sort()
  const until = dates.at(-1) ?? dates[0]
  const workouts = await api.workouts(athleteId, dates[0], until)
  const strengthWorkouts = await api.strengthWorkouts(athleteId, dates[0], until)
  const records = new Map(workouts.map(workout => [String(workout.workoutId), workout]))
  const strengthRecords = new Map(
    strengthWorkouts.map(workout => [`strength:${workout.id}`, workout]),
  )
  const inventory = [
    ...workouts.map(trainingPeaksWorkoutSummary),
    ...strengthWorkouts.map(trainingPeaksStrengthWorkoutSummary),
  ]
  for (const source of sources) {
    const candidates = trainingPeaksTitleCandidates(source, inventory)
    const result: TitleResult = {
      stravaId: source.stravaId,
      date: source.date,
      title: source.title,
      status: 'unmatched',
      candidateWorkoutIds: candidates.map(candidate => candidate.id),
    }
    state.results.push(result)
    if (candidates.length !== 1) {
      result.status = candidates.length ? 'ambiguous' : 'unmatched'
      result.reason = candidates.length
        ? 'Multiple workouts match the recording'
        : 'No matching completed workout'
      continue
    }
    const candidate = candidates[0]
    result.workoutId = candidate.id
    if (
      allSources.filter(other => trainingPeaksTitleCandidates(other, [candidate]).length).length !==
      1
    ) {
      result.status = 'ambiguous'
      result.reason = 'Multiple Strava recordings match this workout'
      continue
    }
    const title = candidate.id.startsWith('strength:')
      ? source.title.trim()
      : trainingPeaksSupportedTitle(source.title)
    result.titleAdjusted = title !== source.title
    if (!title) {
      result.status = 'skipped'
      result.reason = 'The title contains only characters TrainingPeaks cannot save'
      continue
    }
    result.status = 'failed'
    const strengthRecord = strengthRecords.get(candidate.id)
    if (strengthRecord) {
      const current = write
        ? await api.strengthWorkout(athleteId, strengthRecord.id)
        : strengthRecord
      const protection = trainingPeaksStrengthTitleProtection(current)
      if (protection) {
        result.status = 'protected'
        result.reason = protection
        continue
      }
      if (
        current.isLocked ||
        !trainingPeaksTitleCandidates(source, [trainingPeaksStrengthWorkoutSummary(current)]).length
      ) {
        result.status = 'skipped'
        result.reason = current.isLocked
          ? 'Workout is locked'
          : 'Workout no longer matches the recording'
        continue
      }
      if (current.title === title) {
        result.status = 'unchanged'
        continue
      }
      result.before = { title: current.title, description: current.instructions }
      result.after = { title, description: current.instructions }
      if (!write) result.status = 'planned'
      else {
        await api.updateStrengthTitle(current, title)
        result.status = 'updated'
        console.log(`[trainingpeaks-titles] updated ${candidate.id}: ${title}`)
      }
      continue
    }
    // Fetch again before constructing a full-record PUT, including any newly edited notes.
    const current = write
      ? await api.workout(athleteId, Number(candidate.id))
      : records.get(candidate.id)
    if (!current) throw new Error(`TrainingPeaks workout ${candidate.id} is missing`)
    const protection = trainingPeaksTitleProtection(current)
    if (protection) {
      result.status = 'protected'
      result.reason = protection
      continue
    }
    const otherSource = [
      ...(current.description ?? '').matchAll(/strava\.com\/activities\/(\d+)/g),
    ].some(match => Number(match[1]) !== source.stravaId)
    if (
      current.isLocked ||
      otherSource ||
      !trainingPeaksTitleCandidates(source, [trainingPeaksWorkoutSummary(current)]).length
    ) {
      result.status = 'skipped'
      result.reason = current.isLocked
        ? 'Workout is locked'
        : otherSource
          ? 'Description links to another Strava recording'
          : 'Workout no longer matches the recording'
      continue
    }
    const description =
      source.description == null
        ? current.description
        : trainingPeaksSaunaDescription(
            { stravaId: source.stravaId, description: source.description },
            current.description ?? '',
          )
    if (current.title === title && current.description === description) {
      result.status = 'unchanged'
      continue
    }
    result.before = { title: current.title, description: current.description }
    result.after = { title, description }
    if (!write) result.status = 'planned'
    else {
      await api.updateMetadata(current, title, description)
      result.status = 'updated'
      console.log(`[trainingpeaks-titles] updated ${candidate.id}: ${title}`)
    }
  }
}

async function main(argv: readonly string[]): Promise<void> {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(
      'usage: pnpm trainingpeaks:titles [--write | --dry-run | --sources] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--id STRAVA_ID] [--limit N]',
    )
    console.log(
      'Syncs Strava titles for matched completed, unplanned activities. Protects planned workouts and authored instructions, including structured strength. Removes unsupported emoji from fitness titles and reports each adjustment. Retains the existing sauna description sync for unplanned sessions. Defaults to a dry run. Set TRAININGPEAKS_AUTH_COOKIE or TRAININGPEAKS_ACCESS_TOKEN in .env; --sources requires no authentication.',
    )
    return
  }
  const args = parseTrainingPeaksTitleArgs(argv)
  const cache = await readStravaCacheFile(resolve(CACHE, 'strava.json'))
  if (!cache) throw new Error('Strava cache is missing. Run pnpm strava:sync first.')
  const tracking = trainingPeaksSaunaTracking(
    await fs.readFile(resolve(ROOT, 'content/triathlon.md'), 'utf8'),
  )
  const allSources = selectTrainingPeaksTitleSources(cache, tracking)
  const sources = allSources
    .filter(
      source =>
        (!args.since || source.date >= args.since) &&
        (!args.until || source.date <= args.until) &&
        (!args.ids.size || args.ids.has(source.stravaId)),
    )
    .slice(0, args.limit ?? Infinity)
  if (args.sourcesOnly) {
    console.log(JSON.stringify(sources, null, 2))
    return
  }
  console.log(
    `[trainingpeaks-titles] ${sources.length} Strava activities; ${args.write ? 'write' : 'dry run'}`,
  )
  if (!sources.length) return
  const api = new TrainingPeaksApi({
    accessToken: process.env.TRAININGPEAKS_ACCESS_TOKEN,
    authCookie: process.env.TRAININGPEAKS_AUTH_COOKIE,
  })
  const state: TitleRun = { results: [] }
  await syncTitles(api, sources, allSources, args.write, state).catch((error: unknown) => {
    state.error = error instanceof Error ? error.message : String(error)
  })
  const output = resolve(
    CACHE,
    'trainingpeaks-titles',
    `${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
  )
  await fs.mkdir(dirname(output), { recursive: true, mode: 0o700 })
  await fs.writeFile(
    output,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        sourceLastSync: cache.lastSync,
        write: args.write,
        transport: 'trainingpeaks-web-api',
        command: ['pnpm', 'trainingpeaks:titles', ...argv],
        selectedSourceCount: sources.length,
        ...state,
      },
      null,
      2,
    ),
    { mode: 0o600 },
  )
  for (const result of state.results)
    console.log(
      `[trainingpeaks-titles] ${result.date} ${result.status}: ${result.title}${result.reason ? ` (${result.reason})` : ''}`,
    )
  console.log(`[trainingpeaks-titles] report: ${output}`)
  if (state.error) throw new Error(state.error)
  const counts = new Map<string, number>()
  for (const result of state.results)
    counts.set(result.status, (counts.get(result.status) ?? 0) + 1)
  console.log(
    `[trainingpeaks-titles] ${[...counts].map(([status, count]) => `${status}=${count}`).join(' ')}`,
  )
  if (state.results.some(result => ['ambiguous', 'skipped'].includes(result.status)))
    console.warn('[trainingpeaks-titles] Unresolved matches were left unchanged; see the report.')
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch((error: unknown) => {
    console.error(
      `[trainingpeaks-titles] ${error instanceof Error ? error.message : String(error)}`,
    )
    process.exitCode = 1
  })
}
