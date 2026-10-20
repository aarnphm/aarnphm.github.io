import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { allSettledOrThrow, withStagedOutput } from './staged-output'

test('withStagedOutput recovers cleanly if a stage directory for the current pid already exists', async t => {
  const root = await fs.mkdtemp(path.join(tmpdir(), 'quartz-staged-output-eexist-'))
  t.after(async () => {
    await fs.rm(root, { recursive: true, force: true })
  })

  const output = path.join(root, 'public')
  await fs.mkdir(output, { recursive: true })
  await fs.writeFile(path.join(output, 'index.html'), '<h1>initial</h1>\n')

  // Simulate a leftover stage directory for the current pid from an interrupted build
  const stage = path.join(root, '.quartz-cache', `output-stage-public-${process.pid}`)
  await fs.mkdir(path.join(stage, 'static/scripts/chunks'), { recursive: true })
  await fs.writeFile(path.join(stage, 'static/scripts/chunks/orphan.js'), 'orphan()')

  await withStagedOutput(output, async stageDir => {
    await fs.writeFile(path.join(stageDir, 'index.html'), '<h1>recovered</h1>\n')
  })

  assert.equal(await fs.readFile(path.join(output, 'index.html'), 'utf8'), '<h1>recovered</h1>\n')
  await assert.rejects(fs.stat(path.join(output, 'static/scripts/chunks/orphan.js')), {
    code: 'ENOENT',
  })
})

test('withStagedOutput reuses unchanged staged files across incremental builds and rolls back failed builds', async t => {
  const root = await fs.mkdtemp(path.join(tmpdir(), 'quartz-staged-output-reuse-'))
  t.after(async () => {
    await fs.rm(root, { recursive: true, force: true })
  })

  const output = path.join(root, 'public')
  await fs.mkdir(path.join(output, 'notes'), { recursive: true })
  await fs.writeFile(path.join(output, 'index.html'), '<h1>home v1</h1>\n')
  await fs.writeFile(path.join(output, 'notes/unchanged.html'), '<p>stay put</p>\n')
  await fs.writeFile(path.join(output, 'notes/remove-me.html'), '<p>bye</p>\n')

  let stageUnchangedInodeFirst = 0n
  await withStagedOutput(output, async stageDir => {
    const stat = await fs.stat(path.join(stageDir, 'notes/unchanged.html'), { bigint: true })
    stageUnchangedInodeFirst = stat.ino
    await fs.writeFile(path.join(stageDir, 'index.html'), '<h1>home v2</h1>\n')
  })

  assert.equal(await fs.readFile(path.join(output, 'index.html'), 'utf8'), '<h1>home v2</h1>\n')
  assert.equal(
    await fs.readFile(path.join(output, 'notes/unchanged.html'), 'utf8'),
    '<p>stay put</p>\n',
  )

  // A failed build that mutates stage must not leak into output, and the next build must repair stage
  await assert.rejects(
    withStagedOutput(output, async stageDir => {
      await fs.writeFile(path.join(stageDir, 'index.html'), '<h1>corrupted</h1>\n')
      throw new Error('simulated emitter failure')
    }),
    /simulated emitter failure/,
  )
  assert.equal(await fs.readFile(path.join(output, 'index.html'), 'utf8'), '<h1>home v2</h1>\n')

  let stageUnchangedInodeSecond = 0n
  let repairedIndexInStage = ''
  await withStagedOutput(output, async stageDir => {
    const stat = await fs.stat(path.join(stageDir, 'notes/unchanged.html'), { bigint: true })
    stageUnchangedInodeSecond = stat.ino
    repairedIndexInStage = await fs.readFile(path.join(stageDir, 'index.html'), 'utf8')
    await fs.rm(path.join(stageDir, 'notes/remove-me.html'))
    await fs.writeFile(path.join(stageDir, 'notes/added.html'), '<p>new</p>\n')
  })

  assert.equal(
    stageUnchangedInodeSecond,
    stageUnchangedInodeFirst,
    'expected unchanged file in stage to be reused without re-cloning a new inode',
  )
  assert.equal(repairedIndexInStage, '<h1>home v2</h1>\n')
  assert.equal(await fs.readFile(path.join(output, 'notes/added.html'), 'utf8'), '<p>new</p>\n')
  await assert.rejects(fs.stat(path.join(output, 'notes/remove-me.html')), { code: 'ENOENT' })
})

test('allSettledOrThrow waits for sibling async tasks to finish writing before propagating an error', async t => {
  const root = await fs.mkdtemp(path.join(tmpdir(), 'quartz-settled-stage-'))
  t.after(async () => {
    await fs.rm(root, { recursive: true, force: true })
  })

  const output = path.join(root, 'public')
  await fs.mkdir(output, { recursive: true })
  await fs.writeFile(path.join(output, 'index.html'), '<h1>ok</h1>\n')

  let siblingFinishedBeforeStageExit = false
  let stageExited = false

  await assert.rejects(
    withStagedOutput(output, async stageDir => {
      await allSettledOrThrow([
        Promise.reject(new Error('fast scss failure')),
        (async () => {
          await delay(40)
          await fs.mkdir(path.join(stageDir, 'static/scripts/chunks'), { recursive: true })
          await fs.writeFile(path.join(stageDir, 'static/scripts/chunks/slow.js'), 'export {}')
          siblingFinishedBeforeStageExit = !stageExited
        })(),
      ])
    }).finally(() => {
      stageExited = true
    }),
    /fast scss failure/,
  )

  assert.equal(
    siblingFinishedBeforeStageExit,
    true,
    'expected slow sibling task to settle before withStagedOutput exited',
  )
})
