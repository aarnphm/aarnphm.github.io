#!/usr/bin/env -S tsx
import type { ChildProcessByStdio } from 'node:child_process'
import type { Readable } from 'node:stream'
import { spawn } from 'node:child_process'
import { open, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { parseQuartzDevEvent, splitDevEventLines, type QuartzDevEvent } from '../util/dev-events'
import {
  applyQuartzDevEvent,
  createQuartzManagerState,
  resetQuartzManagerState,
} from '../util/dev-manager-state'
import { stopProcessTree, stopProcessTreeByPid } from '../util/process-tree'

const gitRoot = resolveGitRoot()
const daemonPidFile = path.join(gitRoot, '.dev.pid')
const daemonErrFile = path.join(gitRoot, '.dev.err')
const daemonLogFile = '/tmp/quartz-dev.log'
const daemonChildEnv = 'QUARTZ_DEV_DAEMON_CHILD'
const pollIntervalMs = 500
const useColor = process.stdout.isTTY && process.stderr.isTTY
const RESET = '\x1b[0m'
const labelNames = { main: 'main', quartz: 'quartz', wrangler: 'wrangler' } as const
type Label = keyof typeof labelNames
type ManagedChild = ChildProcessByStdio<null, Readable, Readable> & {
  label: Label
  processGroupId?: number
}
const formattedLabels: Record<Label, string> = formatLabels(labelNames)
let shuttingDown = false
let lifecycleRegistered = false
let pnpmDev: ManagedChild | null = null
let wrangler: ManagedChild | null = null
let rawInputEnabled = false
let pnpmDevRetriesRemaining = 0
const managerState = createQuartzManagerState()
let quartzOutputBuffer = ''
let wranglerStartNotBefore = 0
let wranglerBackoffUntil = 0
let wranglerStopInFlight: Promise<void> | null = null
const DEFAULT_DEV_PORT = 7373
const WS_PORT_OFFSET = 1
const WRANGLER_PORT_OFFSET = 707
const MAX_BASE_PORT = 65535 - WRANGLER_PORT_OFFSET
const SLOW_BUILD_THRESHOLD_MS = 100
const WRANGLER_START_DELAY_MS = 250
const WRANGLER_PORT_BACKOFF_MS = 1000
const WRANGLER_EXIT_BACKOFF_MS = 1500
const WRANGLER_DEV_NAME = process.env.WRANGLER_DEV_NAME ?? 'portfolio-dev'
const ANSI_ESCAPE_PATTERN = new RegExp(String.raw`\u001b\[[0-9;]*m`, 'g')

const runtimeConfig = resolveRuntimeConfig(process.argv.slice(2))
const totalPnpmDevAttempts = runtimeConfig.pnpmDevRetryLimit + 1
pnpmDevRetriesRemaining = runtimeConfig.pnpmDevRetryLimit
process.env.PUBLIC_BASE_URL = runtimeConfig.publicBaseUrl

void main()

function resolveGitRoot(): string {
  const envRoot = process.env.GIT_ROOT
  if (envRoot && envRoot.length > 0) {
    return path.resolve(envRoot)
  }
  const current = path.dirname(fileURLToPath(import.meta.url))
  return path.resolve(current, '..', '..')
}

interface RuntimeConfig {
  port: number
  wsPort: number
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

function resolveRuntimeConfig(argv: string[]): RuntimeConfig {
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
    '16',
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
  const prefix = formattedLabels.main ?? '[main]'
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

function failStartup(message: string): never {
  const prefix = formattedLabels.main ?? '[main]'
  process.stderr.write(`${prefix} ${message}\n`)
  process.exit(1)
}

async function main(): Promise<void> {
  if (runtimeConfig.kill) {
    await killDaemon()
    return
  }
  log(
    'main',
    `using dev port ${runtimeConfig.port}, ws ${runtimeConfig.wsPort}, wrangler ${runtimeConfig.wranglerPort}`,
  )
  if (runtimeConfig.daemon) {
    await startDaemon()
    return
  }
  log('main', `launching pnpm dev (attempt 1/${totalPnpmDevAttempts})`)
  launchPnpmDev()
  registerLifecycle()
  registerCtrlCHandler()
  await manageWranglerLoop()
}

async function startDaemon(): Promise<void> {
  if (process.env[daemonChildEnv] === '1') {
    failStartup('daemon child tried to daemonize itself')
  }
  const existingPid = await readDaemonPid()
  if (existingPid !== null) {
    if (processAlive(existingPid)) {
      log('main', `dev daemon already running (pid ${existingPid})`)
      log('main', `logs: ${daemonLogFile}, errors: ${daemonErrFile}`)
      return
    }
    await rm(daemonPidFile, { force: true })
  }

  const stdout = await open(daemonLogFile, 'a')
  const stderr = await open(daemonErrFile, 'a')
  try {
    const child = spawn(
      process.execPath,
      [
        ...process.execArgv,
        fileURLToPath(import.meta.url),
        ...removeDaemonArgs(process.argv.slice(2)),
      ],
      {
        cwd: gitRoot,
        detached: true,
        env: { ...process.env, [daemonChildEnv]: '1' },
        stdio: ['ignore', stdout.fd, stderr.fd],
      },
    )
    const pid = child.pid
    if (pid === undefined) {
      failStartup('failed to start daemon: child pid is unavailable')
    }
    child.unref()
    await writeFile(daemonPidFile, `${pid}\n`)
    log('main', `started dev daemon (pid ${pid})`)
    log('main', `pid: ${daemonPidFile}`)
    log('main', `logs: ${daemonLogFile}`)
    log('main', `errors: ${daemonErrFile}`)
  } finally {
    await stdout.close()
    await stderr.close()
  }
}

async function killDaemon(): Promise<void> {
  const pid = await readDaemonPid()
  if (pid === null) {
    log('main', `no dev daemon pid file at ${daemonPidFile}`)
    return
  }
  if (!processAlive(pid)) {
    await rm(daemonPidFile, { force: true })
    log('main', `removed stale dev daemon pid file for pid ${pid}`)
    return
  }
  const result = await stopProcessTreeByPid(pid, {
    interruptDelayMs: 15000,
    terminateDelayMs: 3000,
  })
  await rm(daemonPidFile, { force: true })
  if (result === 'not-found') {
    log('main', `dev daemon pid ${pid} was already gone`)
    return
  }
  log('main', `stopped dev daemon pid ${pid}`)
}

async function readDaemonPid(): Promise<number | null> {
  let raw: string
  try {
    raw = await readFile(daemonPidFile, 'utf8')
  } catch (err) {
    if (isNotFoundError(err)) return null
    throw err
  }
  const trimmed = raw.trim()
  const pid = Number(trimmed)
  if (!Number.isSafeInteger(pid) || pid < 1) {
    failStartup(`invalid dev daemon pid file at ${daemonPidFile}: ${trimmed}`)
  }
  return pid
}

function removeDaemonArgs(argv: string[]): string[] {
  return argv.filter(arg => arg !== '--daemon' && arg !== '--bg')
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    if (isErrorCode(err, 'ESRCH')) return false
    if (isErrorCode(err, 'EPERM')) return true
    throw err
  }
}

function isNotFoundError(value: unknown): boolean {
  return isErrorCode(value, 'ENOENT')
}

function isErrorCode(value: unknown, code: string): boolean {
  return typeof value === 'object' && value !== null && 'code' in value && value.code === code
}

function launchPnpmDev(): void {
  resetQuartzManagerState(managerState)
  quartzOutputBuffer = ''
  wranglerStartNotBefore = 0
  wranglerBackoffUntil = 0
  void stopWrangler('stopping wrangler while quartz restarts')
  const attempt = runtimeConfig.pnpmDevRetryLimit - pnpmDevRetriesRemaining + 1
  pnpmDev = startProcess(
    runtimeConfig.pnpmDevArgs,
    'quartz',
    `starting quartz (attempt ${attempt}/${totalPnpmDevAttempts})`,
  )
  const child = pnpmDev
  child.on('exit', (code, signal) => handlePnpmDevExit(child, code, signal))
  child.on('error', err => handlePnpmDevError(child, err))
}

function handlePnpmDevExit(
  child: ManagedChild,
  code: number | null,
  signal: NodeJS.Signals | null,
): void {
  if (pnpmDev === child) {
    pnpmDev = null
  }
  if (shuttingDown) {
    return
  }
  const stream: 'stdout' | 'stderr' = code === 0 && signal === null ? 'stdout' : 'stderr'
  log('main', `pnpm dev exited with code ${code ?? 'null'} signal ${signal ?? 'null'}`, stream)
  if (code === 0 && signal === null) {
    void shutdown(0)
    return
  }
  retryPnpmDev(`exit code ${code ?? 'null'} signal ${signal ?? 'null'}`)
}

function handlePnpmDevError(child: ManagedChild, err: Error): void {
  if (pnpmDev === child) {
    pnpmDev = null
  }
  if (shuttingDown) {
    return
  }
  log('main', `pnpm dev error: ${describeError(err)}`, 'stderr')
  retryPnpmDev('process error')
}

function retryPnpmDev(reason: string): void {
  if (pnpmDevRetriesRemaining === 0) {
    log(
      'main',
      `pnpm dev restart budget exhausted after ${totalPnpmDevAttempts} attempts`,
      'stderr',
    )
    void shutdown(1)
    return
  }
  pnpmDevRetriesRemaining -= 1
  const attempt = totalPnpmDevAttempts - pnpmDevRetriesRemaining
  log(
    'main',
    `retrying pnpm dev after ${reason} (attempt ${attempt}/${totalPnpmDevAttempts}, ${pnpmDevRetriesRemaining} retries left)`,
    'stderr',
  )
  launchPnpmDev()
}

function registerLifecycle(): void {
  if (lifecycleRegistered) return
  lifecycleRegistered = true
  const signals: NodeJS.Signals[] = ['SIGINT', 'SIGTERM', 'SIGHUP']
  for (const sig of signals) {
    process.on(sig, () => {
      void shutdown(0, sig)
    })
  }
  process.on('uncaughtException', err => {
    log('main', `manager encountered error: ${describeError(err)}`, 'stderr')
    void shutdown(1)
  })
  process.on('unhandledRejection', reason => {
    log('main', `manager unhandled rejection: ${describeError(reason)}`, 'stderr')
    void shutdown(1)
  })
}

async function manageWranglerLoop(): Promise<void> {
  while (!shuttingDown) {
    const now = Date.now()
    const canStart =
      wrangler === null &&
      managerState.publicAvailable &&
      managerState.quartz === 'ready' &&
      managerState.wrangler !== 'stopping' &&
      now >= wranglerStartNotBefore &&
      now >= wranglerBackoffUntil
    if (canStart) {
      const portAvailable = await isPortAvailable(runtimeConfig.wranglerPort)
      if (!portAvailable) {
        managerState.wrangler = 'backoff'
        wranglerBackoffUntil = Date.now() + WRANGLER_PORT_BACKOFF_MS
        log(
          'main',
          `wrangler port ${runtimeConfig.wranglerPort} is occupied, waiting before retry`,
          'stderr',
        )
        await delay(pollIntervalMs)
        continue
      }
      managerState.wrangler = 'starting'
      wrangler = startProcess(runtimeConfig.wranglerArgs, 'wrangler')
      const current = wrangler
      wrangler.on('exit', (code, signal) => {
        log(
          'main',
          `wrangler exited with code ${code ?? 'null'} signal ${signal ?? 'null'}`,
          code === 0 && signal === null ? 'stdout' : 'stderr',
        )
        if (wrangler === current) {
          wrangler = null
        }
        if (shuttingDown) return
        if (code === 0 && signal === null) {
          managerState.wrangler = 'stopped'
          return
        }
        if (managerState.publicAvailable && managerState.quartz === 'ready') {
          managerState.wrangler = 'backoff'
          wranglerBackoffUntil = Date.now() + WRANGLER_EXIT_BACKOFF_MS
          return
        }
        managerState.wrangler = 'stopped'
      })
      wrangler.on('error', err => {
        log('main', `failed to launch wrangler dev: ${describeError(err)}`, 'stderr')
        if (wrangler === current) {
          wrangler = null
        }
        managerState.wrangler = 'backoff'
        wranglerBackoffUntil = Date.now() + WRANGLER_EXIT_BACKOFF_MS
      })
    }
    const shouldStop =
      wrangler !== null && (!managerState.publicAvailable || managerState.quartz === 'building')
    if (shouldStop) {
      const reason = !managerState.publicAvailable
        ? 'stopping wrangler while public directory is missing'
        : 'stopping wrangler while public is regenerated'
      await stopWrangler(reason)
    }
    await delay(pollIntervalMs)
  }
}

async function shutdown(code: number, signal?: NodeJS.Signals): Promise<void> {
  if (shuttingDown) return
  shuttingDown = true
  if (signal) {
    log('main', `received ${signal}, shutting down...`)
  }
  if (rawInputEnabled && process.stdin.isTTY) {
    process.stdin.setRawMode(false)
    process.stdin.pause()
    process.stdin.removeListener('data', handleStdin)
    rawInputEnabled = false
  }
  if (wrangler) {
    await stopWrangler('stopping wrangler during shutdown')
  }
  if (pnpmDev) {
    await stopProcessTree(pnpmDev)
    pnpmDev = null
  }
  process.exit(code)
}

function startProcess(args: string[], label: Label, message?: string): ManagedChild {
  log('main', message ?? `starting ${label}`)
  const proc = spawn('pnpm', args, {
    cwd: gitRoot,
    env: {
      ...process.env,
      ...(label === 'wrangler' ? { PUBLIC_BASE_URL: runtimeConfig.publicBaseUrl } : {}),
    },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  }) as ManagedChild
  proc.label = label
  proc.processGroupId = proc.pid ?? undefined
  pipeStream(proc)
  return proc
}

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise(resolve => {
    const server = createServer()
    server.once('error', () => resolve(false))
    server.once('listening', () => {
      server.close(() => resolve(true))
    })
    server.listen(port, '0.0.0.0')
  })
}

function pipeStream(child: ManagedChild): void {
  child.stdout.on('data', chunk => {
    handleChildOutput(child.label, chunk)
    process.stdout.write(formatChunk(child.label, chunk))
  })
  child.stderr.on('data', chunk => {
    handleChildOutput(child.label, chunk)
    process.stderr.write(formatChunk(child.label, chunk))
  })
}

function formatChunk(label: Label, chunk: Buffer): string {
  const lines = chunk.toString()
  const prefix = formattedLabels[label] ?? `[${label}]`
  return lines
    .split(/(?<=\n)/)
    .map(line => (line.length > 0 ? `${prefix} ${line}` : line))
    .join('')
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function requestWranglerStart(delayMs: number): void {
  const target = Date.now() + delayMs
  if (target > wranglerStartNotBefore) {
    wranglerStartNotBefore = target
  }
}

async function stopWrangler(reason: string): Promise<void> {
  if (!wrangler) return
  if (wranglerStopInFlight) {
    await wranglerStopInFlight
    return
  }
  managerState.wrangler = 'stopping'
  log('main', reason)
  const current = wrangler
  const stopPromise = stopProcessTree(current).finally(() => {
    if (wrangler === current) {
      wrangler = null
    }
    if (managerState.wrangler === 'stopping') {
      managerState.wrangler = 'stopped'
    }
    wranglerStopInFlight = null
  })
  wranglerStopInFlight = stopPromise
  await stopPromise
}

function registerCtrlCHandler(): void {
  if (!process.stdin.isTTY || rawInputEnabled) return
  process.stdin.setRawMode(true)
  process.stdin.resume()
  process.stdin.on('data', handleStdin)
  rawInputEnabled = true
}

function handleStdin(data: Buffer): void {
  for (const byte of data) {
    if (byte === 0x03) {
      void shutdown(0, 'SIGINT')
      return
    }
  }
}

function handleChildOutput(label: Label, chunk: Buffer): void {
  if (label === 'wrangler') {
    handleWranglerOutput(chunk)
    return
  }
  if (label !== 'quartz') return
  const parsed = splitDevEventLines(quartzOutputBuffer, chunk.toString())
  quartzOutputBuffer = parsed.rest
  for (const line of parsed.lines) {
    handleQuartzLine(line)
  }
}

function handleWranglerOutput(chunk: Buffer): void {
  const text = stripAnsi(chunk.toString())
  if (text.includes(`Ready on http://0.0.0.0:${runtimeConfig.wranglerPort}`)) {
    managerState.wrangler = 'ready'
  }
}

function handleQuartzLine(line: string): void {
  const event = parseQuartzDevEvent(stripAnsi(line).trim())
  if (!event) return
  handleQuartzEvent(event)
}

function handleQuartzEvent(event: QuartzDevEvent): void {
  const actions = applyQuartzDevEvent(managerState, event, WRANGLER_START_DELAY_MS)
  for (const action of actions) {
    switch (action.type) {
      case 'stop-wrangler':
        void stopWrangler(action.reason)
        break
      case 'schedule-wrangler-start':
        wranglerBackoffUntil = 0
        requestWranglerStart(action.delayMs)
        break
    }
  }
}

function log(label: Label, message: string, stream: 'stdout' | 'stderr' = 'stdout'): void {
  const prefix = formattedLabels[label] ?? `[${label}]`
  const target = stream === 'stdout' ? process.stdout : process.stderr
  target.write(`${prefix} ${message}\n`)
}

function formatLabels<T extends Record<string, string>>(map: T): Record<keyof T, string> {
  const widest = Math.max(...Object.values(map).map(value => value.length))
  const result = {} as Record<keyof T, string>
  for (const [key, name] of Object.entries(map) as Array<[keyof T, string]>) {
    const padded = name.padEnd(widest, ' ')
    const base = `[${padded}]`
    if (!useColor) {
      result[key] = base
      continue
    }
    const color = labelColor(key as Label)
    result[key] = color ? `${color}${base}${RESET}` : base
  }
  return result
}

function describeError(value: unknown): string {
  if (value instanceof Error) {
    return value.stack ?? value.message
  }
  if (typeof value === 'string') {
    return value
  }
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function labelColor(label: Label): string | null {
  switch (label) {
    case 'main':
      return '\x1b[38;5;39m'
    case 'quartz':
      return '\x1b[38;5;110m'
    case 'wrangler':
      return '\x1b[38;5;214m'
    default:
      return null
  }
}

function stripAnsi(value: string): string {
  return value.replace(ANSI_ESCAPE_PATTERN, '')
}
