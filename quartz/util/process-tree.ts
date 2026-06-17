import type { ChildProcess } from 'node:child_process'
import { spawn } from 'node:child_process'

export type ProcessTree = ChildProcess & { processGroupId?: number }

export type StopProcessTreeOptions = { interruptDelayMs?: number; terminateDelayMs?: number }
export type StopProcessGroupResult = 'not-found' | 'stopped'
type ProcessTableRow = { pid: number; ppid: number; pgid: number }

export async function stopProcessTree(
  child: ProcessTree,
  options: StopProcessTreeOptions = {},
): Promise<void> {
  if (child.exitCode !== null || child.signalCode) {
    return
  }
  const interruptDelayMs = options.interruptDelayMs ?? 3000
  const terminateDelayMs = options.terminateDelayMs ?? 2000
  const exitPromise = onceExit(child)

  signalProcessTree(child, 'SIGINT')
  const first = await Promise.race([
    exitPromise.then(() => 'exit' as const),
    delay(interruptDelayMs).then(() => 'timeout' as const),
  ])
  if (first === 'timeout' && child.exitCode === null) {
    signalProcessTree(child, 'SIGTERM')
    const second = await Promise.race([
      exitPromise.then(() => 'exit' as const),
      delay(terminateDelayMs).then(() => 'timeout' as const),
    ])
    if (second === 'timeout' && child.exitCode === null) {
      signalProcessTree(child, 'SIGKILL')
      await exitPromise
    }
  }
}

export async function stopProcessGroup(
  processGroupId: number,
  options: StopProcessTreeOptions = {},
): Promise<StopProcessGroupResult> {
  if (!processGroupOrPidAlive(processGroupId)) {
    return 'not-found'
  }
  const interruptDelayMs = options.interruptDelayMs ?? 3000
  const terminateDelayMs = options.terminateDelayMs ?? 2000

  if (!signalProcessGroup(processGroupId, 'SIGINT')) {
    return 'not-found'
  }
  if (await waitForGroupDead(processGroupId, interruptDelayMs)) {
    return 'stopped'
  }
  if (!signalProcessGroup(processGroupId, 'SIGTERM')) {
    return 'stopped'
  }
  if (await waitForGroupDead(processGroupId, terminateDelayMs)) {
    return 'stopped'
  }
  signalProcessGroup(processGroupId, 'SIGKILL')
  await waitForGroupDead(processGroupId, terminateDelayMs)
  return 'stopped'
}

export async function stopProcessTreeByPid(
  rootPid: number,
  options: StopProcessTreeOptions = {},
): Promise<StopProcessGroupResult> {
  const descendantGroups = await collectDescendantProcessGroups(rootPid)
  const result = await stopProcessGroup(rootPid, options)
  for (const groupId of descendantGroups) {
    if (groupId !== rootPid) {
      await stopProcessGroup(groupId, options)
    }
  }
  return result
}

function signalProcessTree(child: ProcessTree, signal: NodeJS.Signals): void {
  const pid = child.processGroupId
  if (pid) {
    try {
      process.kill(-pid, signal)
      return
    } catch (err) {
      if (isGoneProcessError(err)) return
    }
  }
  child.kill(signal)
}

function signalProcessGroup(processGroupId: number, signal: NodeJS.Signals): boolean {
  try {
    process.kill(-processGroupId, signal)
    return true
  } catch (err) {
    if (!isGoneProcessError(err)) throw err
  }
  try {
    process.kill(processGroupId, signal)
    return true
  } catch (err) {
    if (isGoneProcessError(err)) return false
    throw err
  }
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    if (isGoneProcessError(err)) return false
    if (isPermissionError(err)) return true
    throw err
  }
}

function processGroupAlive(processGroupId: number): boolean {
  try {
    process.kill(-processGroupId, 0)
    return true
  } catch (err) {
    if (isGoneProcessError(err)) return false
    if (isPermissionError(err)) return false
    throw err
  }
}

function processGroupOrPidAlive(processGroupId: number): boolean {
  return processGroupAlive(processGroupId) || processAlive(processGroupId)
}

async function waitForGroupDead(processGroupId: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (!processGroupOrPidAlive(processGroupId)) return true
    await delay(50)
  }
  return !processGroupOrPidAlive(processGroupId)
}

async function collectDescendantProcessGroups(rootPid: number): Promise<Set<number>> {
  const rows = await readProcessTable()
  const children = new Map<number, ProcessTableRow[]>()
  for (const row of rows) {
    const siblings = children.get(row.ppid)
    if (siblings) {
      siblings.push(row)
    } else {
      children.set(row.ppid, [row])
    }
  }
  const groups = new Set<number>()
  const pending = [rootPid]
  for (let index = 0; index < pending.length; index += 1) {
    const current = pending[index]
    const descendants = children.get(current)
    if (!descendants) continue
    for (const descendant of descendants) {
      groups.add(descendant.pgid)
      pending.push(descendant.pid)
    }
  }
  return groups
}

function readProcessTable(): Promise<ProcessTableRow[]> {
  return new Promise((resolve, reject) => {
    const child = spawn('ps', ['-axo', 'pid=,ppid=,pgid='], { stdio: ['ignore', 'pipe', 'pipe'] })
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []

    child.stdout?.on('data', chunk => stdout.push(Buffer.from(chunk)))
    child.stderr?.on('data', chunk => stderr.push(Buffer.from(chunk)))
    child.once('error', reject)
    child.once('close', code => {
      if (code !== 0) {
        reject(new Error(Buffer.concat(stderr).toString('utf8').trim()))
        return
      }
      resolve(parseProcessTable(Buffer.concat(stdout).toString('utf8')))
    })
  })
}

function parseProcessTable(output: string): ProcessTableRow[] {
  const rows: ProcessTableRow[] = []
  for (const line of output.split('\n')) {
    const columns = line.trim().split(/\s+/)
    if (columns.length !== 3) continue
    const pid = Number(columns[0])
    const ppid = Number(columns[1])
    const pgid = Number(columns[2])
    if (Number.isSafeInteger(pid) && Number.isSafeInteger(ppid) && Number.isSafeInteger(pgid)) {
      rows.push({ pid, ppid, pgid })
    }
  }
  return rows
}

function isGoneProcessError(value: unknown): boolean {
  return typeof value === 'object' && value !== null && 'code' in value && value.code === 'ESRCH'
}

function isPermissionError(value: unknown): boolean {
  return typeof value === 'object' && value !== null && 'code' in value && value.code === 'EPERM'
}

function onceExit(child: ChildProcess): Promise<void> {
  return new Promise(resolve => {
    child.once('exit', () => resolve())
  })
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
