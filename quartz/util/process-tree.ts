import type { ChildProcess } from 'node:child_process'

export type ProcessTree = ChildProcess & { processGroupId?: number }

export type StopProcessTreeOptions = { interruptDelayMs?: number; terminateDelayMs?: number }

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

function isGoneProcessError(value: unknown): boolean {
  return typeof value === 'object' && value !== null && 'code' in value && value.code === 'ESRCH'
}

function onceExit(child: ChildProcess): Promise<void> {
  return new Promise(resolve => {
    child.once('exit', () => resolve())
  })
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
