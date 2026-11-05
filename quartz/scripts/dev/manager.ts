#!/usr/bin/env -S tsx
import type { ChildProcessByStdio } from 'node:child_process'
import type { Readable } from 'node:stream'
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { formatQuartzCacheReceipt, parseQuartzCacheReceipt } from '../../util/cache-receipts'
import { COURSE_STUDY_PORT_OFFSET } from '../../util/course-study'
import { parseQuartzDevEvent, splitDevEventLines, type QuartzDevEvent } from '../../util/dev-events'
import {
  applyQuartzDevEvent,
  createQuartzManagerState,
  resetQuartzManagerState,
} from '../../util/dev-manager-state'
import { stopProcessTree } from '../../util/process-tree'
import { startCourseStudyServer } from '../course-study-server'
import { resolveRuntimeConfig } from './config'
import { startDaemon, killDaemon } from './daemon'
import { log, formatChunk, stripAnsi, describeError, type Label } from './log'
import { createModelArchiveWatcher } from './models'
import { createWranglerPreviewFailureDetector } from './wrangler-recovery'

const gitRoot = resolveGitRoot()
const pollIntervalMs = 500
type ManagedChild = ChildProcessByStdio<null, Readable, Readable> & {
  label: Label
  processGroupId?: number
}
let shuttingDown = false
let lifecycleRegistered = false
let pnpmDev: ManagedChild | null = null
let wrangler: ManagedChild | null = null
let rawInputEnabled = false
let pnpmDevRetriesRemaining = 0
const managerState = createQuartzManagerState()
const models = createModelArchiveWatcher(
  gitRoot,
  () => managerState.publicAvailable && managerState.quartz === 'ready',
)
let quartzOutputBuffer = ''
let wranglerStartNotBefore = 0
let wranglerBackoffUntil = 0
let wranglerRecoveryNotBefore = 0
let wranglerStopInFlight: Promise<void> | null = null
const WRANGLER_START_DELAY_MS = 250
const WRANGLER_PORT_BACKOFF_MS = 1000
const WRANGLER_EXIT_BACKOFF_MS = 1500
const publicRoot = path.join(gitRoot, 'public')
const crashReportDir = path.join(gitRoot, '.quartz-cache', 'crash-reports')
mkdirSync(crashReportDir, { recursive: true })
const quartzChildNodeOptions = [
  '--max-old-space-size=16384',
  '--report-on-fatalerror',
  `--report-directory=${crashReportDir}`,
].join(' ')

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
  return path.resolve(current, '..', '..', '..')
}

async function main(): Promise<void> {
  if (runtimeConfig.kill) {
    await killDaemon(gitRoot)
    return
  }
  const ports = []
  if (runtimeConfig.serve) {
    ports.push(`dev port ${runtimeConfig.port}`)
  }
  ports.push(`ws ${runtimeConfig.wsPort}`, `wrangler ${runtimeConfig.wranglerPort}`)
  log('main', `using ${ports.join(', ')}`)
  if (runtimeConfig.daemon) {
    await startDaemon(gitRoot)
    return
  }
  managerState.publicAvailable = await readFile(path.join(publicRoot, 'index.html')).then(
    content => content.length > 0,
    () => false,
  )
  if (managerState.publicAvailable) {
    log('main', 'found published output; starting wrangler while quartz warms')
  }
  log('main', `launching pnpm dev (attempt 1/${totalPnpmDevAttempts})`)
  models.start()
  startCourseStudyServer({
    root: gitRoot,
    port: runtimeConfig.port + COURSE_STUDY_PORT_OFFSET,
    log: (message, error) => log('main', message, error ? 'stderr' : 'stdout'),
  }).unref()
  launchPnpmDev()
  registerLifecycle()
  registerCtrlCHandler()
  await manageWranglerLoop()
}

function launchPnpmDev(): void {
  resetQuartzManagerState(managerState)
  quartzOutputBuffer = ''
  wranglerStartNotBefore = 0
  wranglerBackoffUntil = 0
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
      managerState.wrangler !== 'stopping' &&
      now >= wranglerStartNotBefore &&
      now >= wranglerRecoveryNotBefore &&
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
        if (managerState.publicAvailable) {
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
    if (wrangler !== null && !managerState.publicAvailable) {
      await stopWrangler('stopping wrangler while public directory is missing')
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
  models.dispose()
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
      ...(label === 'wrangler'
        ? {
            PUBLIC_BASE_URL: runtimeConfig.publicBaseUrl,
            // macOS asset watchers can exceed OPEN_MAX and make esbuild spawn fail with EBADF.
            ...(process.platform === 'darwin'
              ? {
                  CHOKIDAR_USEPOLLING: process.env.CHOKIDAR_USEPOLLING ?? '1',
                  CHOKIDAR_INTERVAL: process.env.CHOKIDAR_INTERVAL ?? '1000',
                }
              : {}),
          }
        : {}),
      ...(label === 'quartz'
        ? {
            NODE_OPTIONS: [process.env.NODE_OPTIONS, quartzChildNodeOptions]
              .filter(Boolean)
              .join(' '),
          }
        : {}),
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
  const detectPreviewFailure = createWranglerPreviewFailureDetector()
  child.stdout.on('data', chunk => {
    handleChildOutput(child.label, chunk)
    process.stdout.write(formatChunk(child.label, chunk))
  })
  child.stderr.on('data', chunk => {
    handleChildOutput(child.label, chunk)
    process.stderr.write(formatChunk(child.label, chunk))
    if (
      child === wrangler &&
      !shuttingDown &&
      !wranglerStopInFlight &&
      detectPreviewFailure(chunk.toString())
    ) {
      wranglerRecoveryNotBefore = Date.now() + 60_000
      void stopWrangler('remote preview renewal failed; restarting wrangler after a 60s cooldown')
    }
  })
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
  if (
    managerState.wrangler !== 'stopping' &&
    text.includes(`Ready on http://0.0.0.0:${runtimeConfig.wranglerPort}`)
  ) {
    managerState.wrangler = 'ready'
  }
}

function handleQuartzLine(line: string): void {
  const text = stripAnsi(line).trim()
  const receipt = parseQuartzCacheReceipt(text)
  if (receipt) {
    log('main', formatQuartzCacheReceipt(receipt))
    return
  }
  const event = parseQuartzDevEvent(text)
  if (!event) return
  handleQuartzEvent(event)
}

function handleQuartzEvent(event: QuartzDevEvent): void {
  const actions = applyQuartzDevEvent(managerState, event, WRANGLER_START_DELAY_MS)
  if (event.type === 'build:ready' && event.epoch === managerState.currentEpoch) models.sync()
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
