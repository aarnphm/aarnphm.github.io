import type { ExportAppleMultisportTcxOptions, MultisportDiscipline } from './conversion-model'

interface CliArgs extends ExportAppleMultisportTcxOptions {
  sports: MultisportDiscipline[]
  swimDistanceM?: number
  swimDurationS?: number
  swimElapsedTimeS?: number
  swimRoutePath?: string
  transition1DurationS?: number
  transition1RoutePath?: string
  includeTransitions: boolean
}

function usage(): string {
  return [
    'usage: tsx quartz/scripts/export-apple-multisport-tcx.ts --input FILE --id WORKOUT_UUID --output DIRECTORY [--sports swim,bike,run] [--swim-distance-m METERS] [--swim-duration-s SECONDS] [--swim-elapsed-s SECONDS] [--swim-route FILE] [--include-transitions] [--transition-1-duration-s SECONDS] [--transition-1-route FILE]',
    '',
    'writes standalone Strava-targeted TCX files and manifest.json.',
  ].join('\n')
}

function argumentValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

function parseSports(value: string): MultisportDiscipline[] {
  const sports: MultisportDiscipline[] = []
  for (const raw of value.split(',')) {
    const sport = raw.trim()
    if (sport !== 'swim' && sport !== 'bike' && sport !== 'run')
      throw new Error(`--sports accepts swim,bike,run, got ${value}`)
    if (!sports.includes(sport)) sports.push(sport)
  }
  if (sports.length === 0) throw new Error('--sports requires at least one sport')
  return sports
}

function positiveNumber(value: string, flag: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0)
    throw new Error(`${flag} requires a positive finite number`)
  return parsed
}

export function parseAppleMultisportTcxArgs(argv: string[]): CliArgs {
  let inputPath: string | null = null
  let workoutId: string | null = null
  let outputDir: string | null = null
  let sports: MultisportDiscipline[] = ['swim', 'bike', 'run']
  let swimDistanceM: number | undefined
  let swimDurationS: number | undefined
  let swimElapsedTimeS: number | undefined
  let swimRoutePath: string | undefined
  let transition1DurationS: number | undefined
  let transition1RoutePath: string | undefined
  let includeTransitions = false
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]
    if (argument === '--') continue
    if (argument === '--input') inputPath = argumentValue(argv, ++index, argument)
    else if (argument === '--id') workoutId = argumentValue(argv, ++index, argument)
    else if (argument === '--output') outputDir = argumentValue(argv, ++index, argument)
    else if (argument === '--sports') sports = parseSports(argumentValue(argv, ++index, argument))
    else if (argument === '--swim-distance-m')
      swimDistanceM = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--swim-duration-s')
      swimDurationS = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--swim-elapsed-s')
      swimElapsedTimeS = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--swim-route') swimRoutePath = argumentValue(argv, ++index, argument)
    else if (argument === '--transition-1-duration-s')
      transition1DurationS = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--transition-1-route')
      transition1RoutePath = argumentValue(argv, ++index, argument)
    else if (argument === '--include-transitions') includeTransitions = true
    else if (argument === '--help' || argument === '-h') throw new Error(usage())
    else throw new Error(`unknown argument ${argument}\n${usage()}`)
  }
  if (!inputPath || !workoutId || !outputDir) throw new Error(usage())
  return {
    inputPath,
    workoutId,
    outputDir,
    sports,
    ...(swimDistanceM == null ? {} : { swimDistanceM }),
    ...(swimDurationS == null ? {} : { swimDurationS }),
    ...(swimElapsedTimeS == null ? {} : { swimElapsedTimeS }),
    ...(swimRoutePath == null ? {} : { swimRoutePath }),
    ...(transition1DurationS == null ? {} : { transition1DurationS }),
    ...(transition1RoutePath == null ? {} : { transition1RoutePath }),
    includeTransitions,
  }
}
