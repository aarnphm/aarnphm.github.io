import { spawn } from 'node:child_process'
import { open, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { stopProcessTreeByPid } from '../../util/process-tree'
import { failStartup } from './config'
import { log } from './log'

const daemonLogFile = '/tmp/quartz-dev.log'
const daemonChildEnv = 'QUARTZ_DEV_DAEMON_CHILD'

export async function startDaemon(gitRoot: string): Promise<void> {
  const daemonErrFile = path.join(gitRoot, '.dev.err')

  const daemonPidFile = path.join(gitRoot, '.dev.pid')

  if (process.env[daemonChildEnv] === '1') {
    failStartup('daemon child tried to daemonize itself')
  }
  const existingPid = await readDaemonPid(gitRoot)
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
        fileURLToPath(new URL('../dev.ts', import.meta.url)),
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

export async function killDaemon(gitRoot: string): Promise<void> {
  const daemonPidFile = path.join(gitRoot, '.dev.pid')

  const pid = await readDaemonPid(gitRoot)
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

async function readDaemonPid(gitRoot: string): Promise<number | null> {
  const daemonPidFile = path.join(gitRoot, '.dev.pid')

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
