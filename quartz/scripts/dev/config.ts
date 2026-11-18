import { mainPrefix } from './log'

const DEFAULT_DEV_PORT = 7373
const WS_PORT_OFFSET = 1
const WRANGLER_PORT_OFFSET = 707
const MAX_BASE_PORT = 65535 - WRANGLER_PORT_OFFSET
const SLOW_BUILD_THRESHOLD_MS = 100
const WRANGLER_DEV_NAME = process.env.WRANGLER_DEV_NAME ?? 'portfolio-dev'

interface RuntimeConfig {
  port: number
  wsPort: number
  serve: boolean
  wranglerPort: number
  pnpmDevArgs: string[]
  wranglerArgs: string[]
  publicBaseUrl: string
  pnpmDevRetryLimit: number
  daemon: boolean
  kill: boolean
}

interface CliOptions {
  port: number | null
  help: boolean
  retry: number | null
  force: boolean
  serve: boolean
  daemon: boolean
  kill: boolean
  allBuildLogs: boolean
}

function assertBasePort(candidate: number): void {
  if (!Number.isInteger(candidate)) {
    failStartup(`port must be an integer between 1 and ${MAX_BASE_PORT}, got ${candidate}`)
  }
  if (candidate < 1 || candidate > MAX_BASE_PORT) {
    failStartup(`port must be between 1 and ${MAX_BASE_PORT}, got ${candidate}`)
  }
}

export function resolveRuntimeConfig(argv: string[]): RuntimeConfig {
  const { port, help, retry, force, serve, daemon, kill, allBuildLogs } = parseCliOptions(argv)
  if (help) {
    printHelp()
    process.exit(0)
  }
  const envBaseUrl = process.env.PUBLIC_BASE_URL
  const envPort = parsePortFromBase(envBaseUrl)
  const effectivePort = port ?? envPort ?? DEFAULT_DEV_PORT
  assertBasePort(effectivePort)
  const wsPort = effectivePort + WS_PORT_OFFSET
  const wranglerPort = effectivePort + WRANGLER_PORT_OFFSET
  const publicBaseUrl =
    port !== null
      ? `http://localhost:${effectivePort}`
      : (envBaseUrl ?? `http://localhost:${effectivePort}`)
  const pnpmDevArgs = [
    'exec',
    'quartz/bootstrap-cli.mjs',
    'build',
    '--concurrency',
    '6',
    serve ? '--serve' : '--watch',
    '--port',
    String(effectivePort),
    '--wsPort',
    String(wsPort),
  ]
  if (allBuildLogs) {
    pnpmDevArgs.push('--verbose')
  } else {
    pnpmDevArgs.push('--slowBuildThreshold', String(SLOW_BUILD_THRESHOLD_MS))
  }
  if (force) {
    pnpmDevArgs.push('--force')
  }
  const wranglerArgs = [
    'wrangler',
    'dev',
    '--name',
    WRANGLER_DEV_NAME,
    '--port',
    String(wranglerPort),
  ]
  const pnpmDevRetryLimit = retry ?? 3
  return {
    port: effectivePort,
    wsPort,
    serve,
    wranglerPort,
    pnpmDevArgs,
    wranglerArgs,
    publicBaseUrl,
    pnpmDevRetryLimit,
    daemon,
    kill,
  }
}

function parseCliOptions(argv: string[]): CliOptions {
  let port: number | null = null
  let help = false
  let force = false
  let retry: number | null = null
  let serve = false
  let daemon = false
  let kill = false
  let allBuildLogs = false
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (token === '--') {
      continue
    }
    if (token === '--help' || token === '-h') {
      help = true
      continue
    }
    if (token === '--port') {
      const value = argv[index + 1]
      if (!value) {
        failStartup('missing port value for --port')
      }
      index += 1
      port = parsePortValue(value)
      continue
    }
    if (token.startsWith('--port=')) {
      port = parsePortValue(token.slice('--port='.length))
      continue
    }
    if (token === '--retry') {
      const value = argv[index + 1]
      if (!value) {
        failStartup('missing retry value for --retry')
      }
      index += 1
      retry = parseRetryValue(value)
      continue
    }
    if (token.startsWith('--retry=')) {
      retry = parseRetryValue(token.slice('--retry='.length))
    }
    if (token === '--force') {
      force = true
    }
    if (token === '--serve') {
      serve = true
    }
    if (token === '--daemon' || token === '--bg') {
      daemon = true
    }
    if (token === '--kill') {
      kill = true
    }
    if (token === '--allBuildLogs') {
      allBuildLogs = true
    }
  }
  return { port, help, retry, force, serve, daemon, kill, allBuildLogs }
}

function parsePortValue(raw: string): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) {
    failStartup(`invalid port: ${raw}`)
  }
  assertBasePort(parsed)
  return parsed
}

function parseRetryValue(raw: string): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
    failStartup(`invalid retry count: ${raw}`)
  }
  return parsed
}

function printHelp(): void {
  const prefix = mainPrefix
  const lines = [
    'usage: pnpm swarm -- [options]',
    '',
    'options:',
    '  --port <port>       bind pnpm dev to <port>, websocket to <port+1>, wrangler dev to <port+2>',
    '  --retry <count>     restart pnpm dev up to <count> times when it exits non-zero',
    '  --force             enforce running all plugins (longer to compile)',
    '  --allBuildLogs      print all Quartz verbose build logs instead of only slow spans',
    '  --daemon            run manager in the background',
    '  --bg                alias for --daemon',
    '  --kill              stop the daemon from .dev.pid',
    '  --help, -h          show this message',
    '  --compatibility-fixtures <directory>  run the isolated synthetic fixture preview',
    '',
    'environment:',
    '  PUBLIC_BASE_URL     overrides default http://localhost:<port> when --port is not passed',
    '  WRANGLER_DEV_NAME   overrides wrangler dev --name, default portfolio-dev',
    '',
    'example:',
    '  pnpm swarm -- --port 8081',
  ]
  for (const line of lines) {
    process.stdout.write(`${prefix} ${line}\n`)
  }
}

function parsePortFromBase(value?: string): number | null {
  if (!value) {
    return null
  }
  try {
    const candidate = value.includes('://') ? value : `http://${value}`
    const url = new URL(candidate)
    if (!url.port) {
      return null
    }
    const parsed = Number(url.port)
    return Number.isInteger(parsed) ? parsed : null
  } catch {
    return null
  }
}

export function failStartup(message: string): never {
  const prefix = mainPrefix
  process.stderr.write(`${prefix} ${message}\n`)
  process.exit(1)
}
