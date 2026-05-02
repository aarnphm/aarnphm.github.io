import crypto from 'crypto'
import type { QuartzPluginData } from '../plugins/vfile'

export interface EncryptedPayload {
  ciphertext: string
  salt: string
  iv: string
}

export function encryptContent(htmlString: string, password: string): EncryptedPayload {
  const salt = crypto.randomBytes(16)
  const iv = crypto.randomBytes(12)

  const key = crypto.pbkdf2Sync(password, salt, 100000, 32, 'sha256')

  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv)
  let encrypted = cipher.update(htmlString, 'utf8')
  encrypted = Buffer.concat([encrypted, cipher.final()])

  const authTag = cipher.getAuthTag()
  const ciphertext = Buffer.concat([encrypted, authTag])

  const toBase64Url = (buffer: Buffer): string =>
    buffer.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '')

  return { ciphertext: toBase64Url(ciphertext), salt: toBase64Url(salt), iv: toBase64Url(iv) }
}

export function resolveProtectedPassword(data: QuartzPluginData): string {
  const frontmatter = data.frontmatter

  if (frontmatter?.password) {
    const customPassword = process.env[frontmatter.password as string]
    if (customPassword) {
      return customPassword
    }
  }

  const defaultPassword = process.env.PROTECTED_CONTENT_PASSWORD
  if (defaultPassword) {
    return defaultPassword
  }

  throw new Error(
    `No password found for protected content ${data.slug}. ` +
      `Set ${frontmatter?.password || 'PROTECTED_CONTENT_PASSWORD'} environment variable.`,
  )
}
