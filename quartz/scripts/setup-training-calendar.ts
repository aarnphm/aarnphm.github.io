import { isCancel, password } from '@clack/prompts'
import { randomBytes } from 'node:crypto'
import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises'
import { parseEnv } from 'node:util'
import { hashTrainingPassword, validTrainingDataKey } from '../util/training-calendar-crypto'
import { isRecord } from '../util/type-guards'

async function envFile(file: string): Promise<string | null> {
  try {
    return await readFile(file, 'utf8')
  } catch (error) {
    if (isRecord(error) && error.code === 'ENOENT') return null
    throw error
  }
}

async function main(): Promise<void> {
  if (process.argv.includes('--help')) {
    console.log(
      'pnpm exec tsx quartz/scripts/setup-training-calendar.ts\n\nReads TRAININGPEAKS_CALENDAR_PASSWORD from the process environment, then .env, or prompts without echoing it when unset. Saves its scrypt hash and a random encryption key in ignored .env and an existing .dev.vars file, preserving other keys. Retains an existing encryption key. Writes only these two secrets to .quartz-cache/training-calendar-secrets.json for an explicit Wrangler secrets upload. No network requests or deployment. Never copies the plaintext password to generated secrets.',
    )
    return
  }
  const files = await Promise.all(
    ['.env', '.dev.vars'].map(async file => ({ file, content: await envFile(file) })),
  )
  const settings = parseEnv(files[0].content ?? '')
  let entered =
    process.env.TRAININGPEAKS_CALENDAR_PASSWORD ?? settings.TRAININGPEAKS_CALENDAR_PASSWORD
  if (entered === undefined) {
    if (!process.stdin.isTTY)
      throw new Error('Set TRAININGPEAKS_CALENDAR_PASSWORD or run in an interactive terminal.')
    const prompted = await password({
      message: 'Calendar password (15–128 characters)',
      validate(value) {
        const length = [...(value ?? '')].length
        return length < 15 || length > 128
          ? 'Use a unique password of 15–128 characters.'
          : undefined
      },
    })
    if (isCancel(prompted)) return
    const confirmed = await password({ message: 'Repeat the calendar password' })
    if (isCancel(confirmed)) return
    if (prompted !== confirmed) throw new Error('Passwords did not match. No files changed.')
    entered = prompted
  }
  const existingKey = settings.TRAINING_CALENDAR_DATA_KEY
  if (existingKey && !validTrainingDataKey(existingKey))
    throw new Error('Existing TRAINING_CALENDAR_DATA_KEY is invalid. No files changed.')
  const secrets = {
    TRAINING_CALENDAR_PASSWORD_HASH: hashTrainingPassword(entered),
    TRAINING_CALENDAR_DATA_KEY: existingKey ?? randomBytes(32).toString('hex'),
  }
  for (const { file, content } of files) {
    // Creating .dev.vars would stop Wrangler loading the other secrets from .env.
    if (file === '.dev.vars' && content === null) continue
    let next = content ?? ''
    for (const [key, value] of Object.entries(secrets)) {
      const pattern = new RegExp(`^${key}=.*$`, 'gm')
      const line = `${key}='${value}'`
      next = pattern.test(next) ? next.replace(pattern, () => line) : `${next.trimEnd()}\n${line}\n`
    }
    await writeFile(file, next, { mode: 0o600 })
    await chmod(file, 0o600)
  }
  await mkdir('.quartz-cache', { recursive: true, mode: 0o700 })
  await writeFile('.quartz-cache/training-calendar-secrets.json', JSON.stringify(secrets), {
    mode: 0o600,
  })
  await chmod('.quartz-cache/training-calendar-secrets.json', 0o600)
  console.log(
    'Saved the hash and encryption key locally. No secrets were printed and nothing was uploaded.\nThe build must load .env. See docs/training-calendar-access.md for deployment and verification.',
  )
}

main().catch(error => {
  console.error(error instanceof Error ? error.message : 'Calendar setup failed.')
  process.exitCode = 1
})
