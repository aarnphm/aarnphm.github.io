import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { build } from 'esbuild'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { visit } from 'unist-util-visit'
import { parseTrackingBlock, type ManualSaunaEntry } from '../plugins/stores/tracking'
import { readStravaCacheFile } from '../util/strava-cache-file'
import { selectTrainingPeaksSaunaSources } from '../util/trainingpeaks-title-sync'
import type { TrainingPeaksTitleRun } from './trainingpeaks-title-browser'

const exec = promisify(execFile)
const ROOT = resolve(import.meta.dirname, '../..')
const CACHE = resolve(ROOT, 'quartz/.quartz-cache')

interface Args {
  write: boolean
  sourcesOnly: boolean
  browser: string
  since: string | null
  until: string | null
  ids: Set<number>
  limit: number | null
}

export function parseTrainingPeaksTitleArgs(argv: readonly string[]): Args {
  const args: Args = { write: false, sourcesOnly: false, browser: process.env.TRAININGPEAKS_BROWSER || 'Helium', since: null, until: null, ids: new Set(), limit: null }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--') continue
    if (arg === '--write') args.write = true
    else if (arg === '--dry-run') args.write = false
    else if (arg === '--sources') args.sourcesOnly = true
    else if (['--browser', '--since', '--until', '--id', '--limit'].includes(arg)) {
      const value = argv[++i]
      if (!value || value.startsWith('--')) throw new Error(`${arg} requires a value`)
      if (arg === '--browser') args.browser = value
      else if (arg === '--since' || arg === '--until') {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString().slice(0, 10) !== value)
          throw new Error(`${arg} requires a valid YYYY-MM-DD date`)
        if (arg === '--since') args.since = value
        else args.until = value
      } else {
        const number = Number(value)
        if (!Number.isSafeInteger(number) || number <= 0) throw new Error(`${arg} requires a positive integer`)
        if (arg === '--id') args.ids.add(number)
        else args.limit = number
      }
    } else throw new Error(`Unknown TrainingPeaks title argument: ${arg}`)
  }
  if (args.since && args.until && args.since > args.until) throw new Error('--since must precede --until')
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

const APPLESCRIPT = `on run argv
  set browserName to item 1 of argv
  set javascriptSource to item 2 of argv
  using terms from application "Google Chrome"
    tell application browserName
      if not running then error "Open the signed-in TrainingPeaks calendar first."
      set matches to {}
      repeat with browserWindow in windows
        repeat with browserTab in tabs of browserWindow
          if URL of browserTab starts with "https://app.trainingpeaks.com/#calendar" then set end of matches to browserTab
        end repeat
      end repeat
      if (count matches) is 0 then error "Open the signed-in TrainingPeaks calendar first."
      if (count matches) is not 1 then error "Keep exactly one TrainingPeaks calendar tab open for this command."
      return execute item 1 of matches javascript javascriptSource
    end tell
  end using terms from
end run`

async function browserJavaScript(browser: string, source: string): Promise<string> {
  const { stdout } = await exec('osascript', ['-e', APPLESCRIPT, browser, source], { timeout: 20_000, maxBuffer: 4 * 1024 * 1024 })
  return stdout.trim()
}

export async function trainingPeaksTitleBrowserBundle(): Promise<string> {
  const result = await build({
    entryPoints: [resolve(import.meta.dirname, 'trainingpeaks-title-browser.ts')],
    bundle: true,
    write: false,
    format: 'iife',
    globalName: 'gardenTrainingPeaksTitleSync',
    platform: 'browser',
    target: 'es2022',
  })
  const output = result.outputFiles[0]?.text
  if (!output) throw new Error('Could not bundle the TrainingPeaks calendar sync')
  return output
}

async function main(argv: readonly string[]): Promise<void> {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log('usage: pnpm trainingpeaks:titles [--write | --dry-run | --sources] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--id STRAVA_ID] [--limit N] [--browser Helium|Google Chrome]')
    console.log('Syncs recorded Strava sauna titles and descriptions to matching completed TrainingPeaks cardio workouts. Defaults to a dry run. Open the relevant dates in the signed-in TrainingPeaks calendar. The macOS browser must already allow JavaScript from Apple Events.')
    return
  }
  const args = parseTrainingPeaksTitleArgs(argv)
  const cache = await readStravaCacheFile(resolve(CACHE, 'strava.json'))
  if (!cache) throw new Error('Strava cache is missing. Run pnpm strava:sync first.')
  const tracking = trainingPeaksSaunaTracking(await fs.readFile(resolve(ROOT, 'content/triathlon.md'), 'utf8'))
  const sources = selectTrainingPeaksSaunaSources(cache, tracking).filter(source =>
    (!args.since || source.date >= args.since) &&
    (!args.until || source.date <= args.until) &&
    (!args.ids.size || args.ids.has(source.stravaId)),
  ).slice(0, args.limit ?? Infinity)
  if (args.sourcesOnly) {
    console.log(JSON.stringify(sources, null, 2))
    return
  }
  console.log(`[trainingpeaks-titles] ${sources.length} Strava sauna recordings; ${args.write ? 'write' : 'dry run'}`)
  if (!sources.length) return
  if (process.platform !== 'darwin') throw new Error('TrainingPeaks calendar sync currently requires macOS and a Chromium browser')
  await browserJavaScript(args.browser, `${await trainingPeaksTitleBrowserBundle()}\ngardenTrainingPeaksTitleSync.startTrainingPeaksTitleSync(${JSON.stringify(sources)}, ${args.write}); 'started'`)
  const deadline = Date.now() + 240_000
  let state: TrainingPeaksTitleRun
  do {
    await new Promise(resolve => setTimeout(resolve, 500))
    state = JSON.parse(await browserJavaScript(args.browser, 'JSON.stringify(window.gardenTrainingPeaksTitles)'))
    if (Date.now() > deadline) throw new Error('TrainingPeaks sync exceeded four minutes; inspect the open calendar before retrying')
  } while (state.running)
  const output = resolve(CACHE, 'trainingpeaks-titles', `${new Date().toISOString().replace(/[:.]/g, '-')}.json`)
  await fs.mkdir(dirname(output), { recursive: true, mode: 0o700 })
  await fs.writeFile(output, JSON.stringify({ generatedAt: new Date().toISOString(), sourceLastSync: cache.lastSync, write: args.write, ...state }, null, 2), { mode: 0o600 })
  for (const result of state.results) console.log(`[trainingpeaks-titles] ${result.date} ${result.status}: ${result.title}${result.reason ? ` (${result.reason})` : ''}`)
  console.log(`[trainingpeaks-titles] report: ${output}`)
  if (state.error) throw new Error(state.error)
  if (state.results.some(result => ['unmatched', 'ambiguous', 'skipped'].includes(result.status))) process.exitCode = 1
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch((error: unknown) => {
    console.error(`[trainingpeaks-titles] ${error instanceof Error ? error.message : String(error)}`)
    process.exitCode = 1
  })
}
