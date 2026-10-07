// Server/build only. Never import this module into browser code.
import { Buffer } from 'node:buffer'
import {
  createCipheriv,
  createDecipheriv,
  randomBytes,
  scryptSync,
  timingSafeEqual,
} from 'node:crypto'
import { isRecord } from './type-guards'

const scryptOptions = { N: 32768, r: 8, p: 3, maxmem: 64 * 1024 * 1024 }
const passwordPattern = /^scrypt\$32768\$8\$3\$([a-f0-9]{32})\$([a-f0-9]{64})$/
const aad = Buffer.from('garden:training-calendar:v1')

export function validTrainingDataKey(key: string | undefined): key is string {
  return typeof key === 'string' && /^[a-f0-9]{64}$/.test(key)
}

export function validTrainingPasswordHash(hash: string | undefined): hash is string {
  return typeof hash === 'string' && passwordPattern.test(hash)
}

export function hashTrainingPassword(password: string): string {
  if ([...password].length < 15 || [...password].length > 128)
    throw new Error('Use a unique password of 15–128 characters.')
  const salt = randomBytes(16)
  const hash = scryptSync(password, salt, 32, scryptOptions)
  return `scrypt$32768$8$3$${salt.toString('hex')}$${hash.toString('hex')}`
}

export function verifyTrainingPassword(password: string, encoded: string): boolean {
  const match = passwordPattern.exec(encoded)
  if (!match || password.length > 256) return false
  const hash = scryptSync(password, Buffer.from(match[1], 'hex'), 32, scryptOptions)
  return timingSafeEqual(hash, Buffer.from(match[2], 'hex'))
}

/** Missing configuration overwrites any previous plaintext asset with an empty tombstone. */
export function sealTrainingCalendar(calendar: unknown, key: string | undefined): string {
  if (!validTrainingDataKey(key)) return 'null'
  const iv = randomBytes(12)
  const cipher = createCipheriv('aes-256-gcm', Buffer.from(key, 'hex'), iv)
  cipher.setAAD(aad)
  const ciphertext = Buffer.concat([
    cipher.update(JSON.stringify(calendar), 'utf8'),
    cipher.final(),
  ])
  return JSON.stringify({
    v: 1,
    iv: iv.toString('hex'),
    tag: cipher.getAuthTag().toString('hex'),
    ciphertext: ciphertext.toString('base64'),
  })
}

export function openTrainingCalendar(value: unknown, key: string): string {
  if (
    !validTrainingDataKey(key) ||
    !isRecord(value) ||
    value.v !== 1 ||
    typeof value.iv !== 'string' ||
    !/^[a-f0-9]{24}$/.test(value.iv) ||
    typeof value.tag !== 'string' ||
    !/^[a-f0-9]{32}$/.test(value.tag) ||
    typeof value.ciphertext !== 'string'
  )
    throw new Error('Encrypted training calendar unavailable')
  const decipher = createDecipheriv(
    'aes-256-gcm',
    Buffer.from(key, 'hex'),
    Buffer.from(value.iv, 'hex'),
  )
  decipher.setAAD(aad)
  decipher.setAuthTag(Buffer.from(value.tag, 'hex'))
  return Buffer.concat([
    decipher.update(Buffer.from(value.ciphertext, 'base64')),
    decipher.final(),
  ]).toString('utf8')
}
