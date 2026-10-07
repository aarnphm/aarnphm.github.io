import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, before, test } from 'node:test'
import { parseEnv } from 'node:util'
import { validTrainingDataKey, verifyTrainingPassword } from '../util/training-calendar-crypto'

// Run the real CLI with piped stdin: env selection, validation before writes,
// preserved settings/key, and plaintext leaking into generated secrets can fail here.
const processPassword = 'synthetic process password 2026'
const filePassword = 'synthetic file password 2026'
const cli = resolve('node_modules/tsx/dist/cli.mjs')
const script = resolve('quartz/scripts/setup-training-calendar.ts')
let evidence: string

before(async () => {
  evidence = await mkdtemp(join(tmpdir(), 'garden-training-env-'))
  await writeFile(
    join(evidence, 'README.md'),
    'Repeat: `pnpm test quartz/scripts/setup-training-calendar.test.ts`\n\nEach case runs the actual setup CLI with piped stdin in its own directory. All passwords and settings are synthetic. Result files retain the exit status and output; generated files allow inspection of preserved settings, encryption keys, and password verifiers.\n',
  )
})

after(() => console.log(`Training calendar env evidence: ${evidence}`))

async function run(directory: string, suppliedPassword?: string) {
  const env = { ...process.env }
  delete env.TRAININGPEAKS_CALENDAR_PASSWORD
  if (suppliedPassword !== undefined) env.TRAININGPEAKS_CALENDAR_PASSWORD = suppliedPassword
  const result = spawnSync(process.execPath, [cli, script], {
    cwd: directory,
    env,
    encoding: 'utf8',
    timeout: 15_000,
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  assert.ifError(result.error)
  await writeFile(
    `${directory}.result.json`,
    JSON.stringify(
      { status: result.status, stdout: result.stdout, stderr: result.stderr },
      null,
      2,
    ),
  )
  return result
}

for (const source of ['process', 'file', 'process-over-file']) {
  test(`sets up the calendar non-interactively from ${source}`, async () => {
    const directory = join(evidence, source)
    await mkdir(directory)
    const fromFile = source !== 'process'
    const fromProcess = source !== 'file'
    const key = 'ab'.repeat(32)
    await writeFile(
      join(directory, '.env'),
      `UNCHANGED_SENTINEL='keep-me'\nTRAINING_CALENDAR_DATA_KEY='${key}'\n` +
        (fromFile ? `TRAININGPEAKS_CALENDAR_PASSWORD='${filePassword}'\n` : ''),
    )
    await writeFile(join(directory, '.dev.vars'), "DEV_SENTINEL='keep-dev'\n")
    const result = await run(directory, fromProcess ? processPassword : undefined)
    assert.equal(result.status, 0, result.stderr)
    const saved = parseEnv(await readFile(join(directory, '.env'), 'utf8'))
    const dev = parseEnv(await readFile(join(directory, '.dev.vars'), 'utf8'))
    const manifest = await readFile(
      join(directory, '.quartz-cache/training-calendar-secrets.json'),
      'utf8',
    )
    assert.equal(saved.UNCHANGED_SENTINEL, 'keep-me')
    assert.equal(saved.TRAINING_CALENDAR_DATA_KEY, key)
    assert.ok(validTrainingDataKey(saved.TRAINING_CALENDAR_DATA_KEY))
    assert.ok(
      verifyTrainingPassword(
        fromProcess ? processPassword : filePassword,
        saved.TRAINING_CALENDAR_PASSWORD_HASH,
      ),
    )
    assert.equal(dev.DEV_SENTINEL, 'keep-dev')
    assert.equal(dev.TRAINING_CALENDAR_PASSWORD_HASH, saved.TRAINING_CALENDAR_PASSWORD_HASH)
    assert.equal(dev.TRAININGPEAKS_CALENDAR_PASSWORD, undefined)
    assert.deepEqual(JSON.parse(manifest), {
      TRAINING_CALENDAR_PASSWORD_HASH: saved.TRAINING_CALENDAR_PASSWORD_HASH,
      TRAINING_CALENDAR_DATA_KEY: key,
    })
    for (const output of [result.stdout, result.stderr, manifest]) {
      assert.ok(!output.includes(processPassword))
      assert.ok(!output.includes(filePassword))
    }
    assert.equal((await stat(join(directory, '.env'))).mode & 0o777, 0o600)
  })
}

test('invalid environment passwords fail before changing files or falling back', async () => {
  for (const [index, value] of ['', 'short', 'x'.repeat(129)].entries()) {
    const directory = join(evidence, `invalid-${index}`)
    await mkdir(directory)
    const original = `TRAININGPEAKS_CALENDAR_PASSWORD='${filePassword}'\n`
    await writeFile(join(directory, '.env'), original)
    const result = await run(directory, value)
    assert.equal(result.status, 1)
    assert.match(result.stderr, /15–128 characters/)
    assert.equal(await readFile(join(directory, '.env'), 'utf8'), original)
    assert.deepEqual(await readdir(directory), ['.env'])
  }
})

test('missing password with piped stdin reports the environment option without writing files', async () => {
  const directory = join(evidence, 'missing')
  await mkdir(directory)
  const result = await run(directory)
  assert.equal(result.status, 1)
  assert.match(result.stderr, /TRAININGPEAKS_CALENDAR_PASSWORD/)
  assert.deepEqual(await readdir(directory), [])
})
