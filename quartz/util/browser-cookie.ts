import { execFile } from 'node:child_process'
import { createDecipheriv, createHash, pbkdf2Sync } from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { promisify } from 'node:util'
import { isRecord, readString } from './type-guards'

const execFileAsync = promisify(execFile)
const CHROME_EPOCH_DELTA_MS = 11_644_473_600_000n
const GARMIN_HOST = 'connect.garmin.com'

interface BrowserRoot {
  name: string
  root: string
  keychainAccount: string
  keychainService: string
}

interface CookieRecord {
  host: string
  name: string
  value: string
  encryptedValue: Uint8Array
  path: string
  expiresUtc: bigint
}

const browserRoots: readonly BrowserRoot[] = [
  {
    name: 'Helium',
    root: join(os.homedir(), 'Library', 'Application Support', 'net.imput.helium'),
    keychainAccount: 'Helium',
    keychainService: 'Helium Storage Key',
  },
  {
    name: 'Chrome',
    root: join(os.homedir(), 'Library', 'Application Support', 'Google', 'Chrome'),
    keychainAccount: 'Chrome',
    keychainService: 'Chrome Safe Storage',
  },
  {
    name: 'Chromium',
    root: join(os.homedir(), 'Library', 'Application Support', 'Chromium'),
    keychainAccount: 'Chromium',
    keychainService: 'Chromium Safe Storage',
  },
  {
    name: 'Brave',
    root: join(os.homedir(), 'Library', 'Application Support', 'BraveSoftware', 'Brave-Browser'),
    keychainAccount: 'Brave',
    keychainService: 'Brave Safe Storage',
  },
  {
    name: 'Edge',
    root: join(os.homedir(), 'Library', 'Application Support', 'Microsoft Edge'),
    keychainAccount: 'Microsoft Edge',
    keychainService: 'Microsoft Edge Safe Storage',
  },
]

async function exists(path: string): Promise<boolean> {
  try {
    await fs.access(path)
    return true
  } catch {
    return false
  }
}

function hostMatches(cookieHost: string, target: string): boolean {
  const host = cookieHost.startsWith('.') ? cookieHost.slice(1) : cookieHost
  return target === host || target.endsWith(`.${host}`)
}

function isExpired(expiresUtc: bigint): boolean {
  if (expiresUtc <= 0n) return false
  const expiresMs = expiresUtc / 1000n - CHROME_EPOCH_DELTA_MS
  return expiresMs <= BigInt(Date.now())
}

async function profileCookieDbs(root: string): Promise<string[]> {
  const out: string[] = []
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => [])
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const profile = join(root, entry.name)
    const paths = [join(profile, 'Cookies'), join(profile, 'Network', 'Cookies')]
    for (const path of paths) if (await exists(path)) out.push(path)
  }
  return out
}

async function cookieDbCandidates(): Promise<{ browser: BrowserRoot; db: string }[]> {
  const explicit = process.env.GARMIN_CONNECT_COOKIE_DB?.trim()
  if (explicit) return [{ browser: browserRoots[0], db: explicit }]

  const selected = process.env.GARMIN_CONNECT_BROWSER?.trim().toLowerCase()
  const roots = selected
    ? browserRoots.filter(browser => browser.name.toLowerCase() === selected)
    : browserRoots

  const out: { browser: BrowserRoot; db: string }[] = []
  for (const browser of roots) {
    for (const db of await profileCookieDbs(browser.root)) out.push({ browser, db })
  }
  return out
}

function cookieRecord(row: unknown): CookieRecord | null {
  if (!isRecord(row)) return null
  const host = readString(row, 'host_key')
  const name = readString(row, 'name')
  const value = readString(row, 'value') ?? ''
  const encrypted = row.encrypted_value
  const path = readString(row, 'path') ?? '/'
  const expiresUtc = readSqliteInteger(row, 'expires_utc') ?? 0n
  if (!host || !name || !(encrypted instanceof Uint8Array)) return null
  return { host, name, value, encryptedValue: encrypted, path, expiresUtc }
}

function queryCookies(dbPath: string): CookieRecord[] {
  const db = new DatabaseSync(dbPath, { readOnly: true, readBigInts: true })
  try {
    const rows = db
      .prepare(
        "select host_key, name, value, encrypted_value, path, expires_utc from cookies where host_key like '%garmin.com'",
      )
      .all()
    return rows.map(cookieRecord).filter((cookie): cookie is CookieRecord => cookie != null)
  } finally {
    db.close()
  }
}

async function keychainPassword(browser: BrowserRoot): Promise<string | null> {
  if (process.platform !== 'darwin') return null
  try {
    const { stdout } = await execFileAsync('security', [
      'find-generic-password',
      '-a',
      browser.keychainAccount,
      '-s',
      browser.keychainService,
      '-w',
    ])
    const password = stdout.trim()
    return password || null
  } catch {
    return null
  }
}

function readSqliteInteger(record: Record<string, unknown>, key: string): bigint | undefined {
  const value = record[key]
  if (typeof value === 'bigint') return value
  if (typeof value === 'number' && Number.isSafeInteger(value)) return BigInt(value)
  return undefined
}

function stripHostHash(host: string, decrypted: Buffer): Buffer {
  const hash = createHash('sha256').update(host).digest()
  if (decrypted.subarray(0, hash.length).equals(hash)) return decrypted.subarray(hash.length)
  return decrypted
}

function decryptChromeValue(
  host: string,
  encrypted: Uint8Array,
  password: string | null,
): string | null {
  const bytes = Buffer.from(encrypted)
  if (bytes.length === 0) return ''
  const prefix = bytes.subarray(0, 3).toString('ascii')
  if ((prefix !== 'v10' && prefix !== 'v11') || !password) return null
  const key = pbkdf2Sync(password, 'saltysalt', 1003, 16, 'sha1')
  const decipher = createDecipheriv('aes-128-cbc', key, Buffer.alloc(16, ' '))
  const out = Buffer.concat([decipher.update(bytes.subarray(3)), decipher.final()])
  return stripHostHash(host, out).toString('utf8')
}

async function cookieValue(cookie: CookieRecord, browser: BrowserRoot): Promise<string | null> {
  if (cookie.value) return cookie.value
  return decryptChromeValue(cookie.host, cookie.encryptedValue, await keychainPassword(browser))
}

export async function browserCookieHeader(): Promise<string | null> {
  const pairs = new Map<string, string>()
  for (const candidate of await cookieDbCandidates()) {
    let cookies: CookieRecord[]
    try {
      cookies = queryCookies(candidate.db)
        .filter(cookie => hostMatches(cookie.host, GARMIN_HOST))
        .filter(cookie => !isExpired(cookie.expiresUtc))
        .sort((a, b) => a.host.length - b.host.length || a.path.length - b.path.length)
    } catch {
      continue
    }
    for (const cookie of cookies) {
      let value: string | null
      try {
        value = await cookieValue(cookie, candidate.browser)
      } catch {
        continue
      }
      if (value) pairs.set(cookie.name, value)
    }
    if (pairs.size > 0) return [...pairs].map(([name, value]) => `${name}=${value}`).join('; ')
  }
  return null
}
