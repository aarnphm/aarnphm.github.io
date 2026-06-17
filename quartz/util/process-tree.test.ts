import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import process from 'node:process'
import test from 'node:test'
import {
  stopProcessGroup,
  stopProcessTree,
  stopProcessTreeByPid,
  type ProcessTree,
} from './process-tree'

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

async function waitForDead(pid: number): Promise<void> {
  for (let attempt = 0; attempt < 25; attempt += 1) {
    if (!pidAlive(pid)) return
    await new Promise(resolve => setTimeout(resolve, 40))
  }
  assert.equal(pidAlive(pid), false)
}

test('stopProcessTree terminates descendants in the child process group', async () => {
  const script = `
    const { spawn } = require('node:child_process')
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
    process.stdout.write(String(child.pid) + '\\n')
    setInterval(() => {}, 1000)
  `
  const child: ProcessTree = spawn(process.execPath, ['-e', script], {
    detached: true,
    stdio: ['ignore', 'pipe', 'ignore'],
  })
  child.processGroupId = child.pid ?? undefined

  const descendantPid = await new Promise<number>(resolve => {
    child.stdout?.once('data', chunk => resolve(Number(chunk.toString().trim())))
  })
  assert.notEqual(descendantPid, 0)
  assert.equal(pidAlive(descendantPid), true)

  await stopProcessTree(child, { interruptDelayMs: 50, terminateDelayMs: 50 })
  await waitForDead(descendantPid)
})

test('stopProcessGroup terminates a detached process group by pid', async () => {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], {
    detached: true,
    stdio: 'ignore',
  })
  const pid = child.pid
  if (pid === undefined) {
    throw new Error('detached child pid is unavailable')
  }

  assert.equal(pidAlive(pid), true)
  const result = await stopProcessGroup(pid, { interruptDelayMs: 50, terminateDelayMs: 50 })

  assert.equal(result, 'stopped')
  await waitForDead(pid)
})

test('stopProcessTreeByPid terminates detached child groups', async () => {
  const script = `
    const { spawn } = require('node:child_process')
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], {
      detached: true,
      stdio: 'ignore',
    })
    process.stdout.write(String(child.pid) + '\\n')
    setInterval(() => {}, 1000)
  `
  const child = spawn(process.execPath, ['-e', script], {
    detached: true,
    stdio: ['ignore', 'pipe', 'ignore'],
  })
  const pid = child.pid
  if (pid === undefined) {
    throw new Error('root child pid is unavailable')
  }
  const descendantPid = await new Promise<number>(resolve => {
    child.stdout?.once('data', chunk => resolve(Number(chunk.toString().trim())))
  })

  assert.equal(pidAlive(descendantPid), true)
  const result = await stopProcessTreeByPid(pid, { interruptDelayMs: 50, terminateDelayMs: 50 })

  assert.equal(result, 'stopped')
  await waitForDead(pid)
  await waitForDead(descendantPid)
})
