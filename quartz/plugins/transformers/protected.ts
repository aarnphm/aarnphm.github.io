import crypto from "crypto"
import { QuartzTransformerPlugin } from "../types"
import { Root } from "hast"
//@ts-ignore
import script from "../../components/scripts/protected.inline"
import content from "../../components/styles/protected.scss"

interface EncryptedPayload {
  ciphertext: string
  salt: string
  iv: string
}

function getPasswordForPage(file: any): string {
  const frontmatter = file.data.frontmatter

  if (frontmatter?.password) {
    const customPassword = process.env[frontmatter.password]
    if (customPassword) {
      return customPassword
    }
  }

  const defaultPassword = process.env.PROTECTED_CONTENT_PASSWORD
  if (defaultPassword) {
    return defaultPassword
  }

  throw new Error(
    `No password found for protected page ${file.data.slug}. ` +
      `Set ${frontmatter?.password || "PROTECTED_CONTENT_PASSWORD"} environment variable.`,
  )
}

function encryptContent(htmlString: string, password: string): EncryptedPayload {
  // Generate random salt (16 bytes)
  const salt = crypto.randomBytes(16)

  // Generate random IV for AES-GCM (12 bytes)
  const iv = crypto.randomBytes(12)

  // Derive encryption key using PBKDF2
  const key = crypto.pbkdf2Sync(
    password,
    salt,
    100000, // iterations
    32, // key length (256 bits)
    "sha256",
  )

  // Encrypt with AES-256-GCM
  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv)
  let encrypted = cipher.update(htmlString, "utf8")
  encrypted = Buffer.concat([encrypted, cipher.final()])

  // Get authentication tag
  const authTag = cipher.getAuthTag()

  // Combine ciphertext and auth tag
  const ciphertext = Buffer.concat([encrypted, authTag])

  // Use base64url encoding (URL-safe, no padding) to avoid HTML attribute issues
  const toBase64Url = (buffer: Buffer): string => {
    return buffer.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "")
  }

  return {
    ciphertext: toBase64Url(ciphertext),
    salt: toBase64Url(salt),
    iv: toBase64Url(iv),
  }
}

export const Protected: QuartzTransformerPlugin = () => {
  return {
    name: "Protected",
    htmlPlugins: () => [
      () => {
        return async (tree: Root, file) => {
          const frontmatter = file.data.frontmatter
          if (!frontmatter?.protected) return

          const password = getPasswordForPage(file)
          file.data.protectedPassword = password
        }
      },
    ],
    externalResources() {
      return {
        js: [
          {
            loadTime: "afterDOMReady",
            contentType: "inline",
            script,
          },
        ],
        css: [
          {
            content,
            inline: true,
          },
        ],
      }
    },
  }
}
