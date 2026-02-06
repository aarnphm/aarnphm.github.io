interface EncryptedPayload {
  ciphertext: string
  salt: string
  iv: string
}

const unlockTimers = new Map<string, number>()

const unlockKey = (slug: string): string => `protected-until:${slug}`

function readUnlockUntil(slug: string) {
  const raw = window.localStorage.getItem(unlockKey(slug))
  if (!raw) return null

  const unlockUntil = parseInt(raw, 10)
  if (Number.isNaN(unlockUntil)) {
    window.localStorage.removeItem(unlockKey(slug))
    return null
  }

  return unlockUntil
}

function clearUnlockTimer(slug: string) {
  const timerId = unlockTimers.get(slug)
  if (typeof timerId === 'number') {
    window.clearTimeout(timerId)
    unlockTimers.delete(slug)
  }
}

function ensureArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  if (bytes.buffer instanceof ArrayBuffer) {
    if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
      return bytes.buffer
    }
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  }

  const buffer = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(buffer).set(bytes)
  return buffer
}

async function deriveKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const encoder = new TextEncoder()
  const passwordKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    'PBKDF2',
    false,
    ['deriveKey'],
  )

  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: ensureArrayBuffer(salt), iterations: 100000, hash: 'SHA-256' },
    passwordKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['decrypt'],
  )
}

async function decryptContent(encryptedData: EncryptedPayload, password: string): Promise<string> {
  // Convert base64url back to base64 (add padding and restore +/)
  const fromBase64Url = (str: string | undefined): string => {
    if (!str || typeof str !== 'string') {
      throw new Error(`invalid base64url string: ${str}`)
    }
    // Replace URL-safe characters back to standard base64
    let base64 = str.replace(/-/g, '+').replace(/_/g, '/')
    // Add padding
    while (base64.length % 4) {
      base64 += '='
    }
    return base64
  }

  const salt = Uint8Array.from(atob(fromBase64Url(encryptedData.salt)), c => c.charCodeAt(0))
  const iv = Uint8Array.from(atob(fromBase64Url(encryptedData.iv)), c => c.charCodeAt(0))
  const ciphertext = Uint8Array.from(atob(fromBase64Url(encryptedData.ciphertext)), c =>
    c.charCodeAt(0),
  )

  const key = await deriveKey(password, salt)

  try {
    const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext)

    return new TextDecoder().decode(decrypted)
  } catch {
    throw new Error('decryption failed')
  }
}

// TTL for decrypted content in milliseconds (30 minutes)
const DECRYPTION_TTL = 30 * 60 * 1000

function reLockContent(article: Element, slug: string): void {
  const decryptedContent = article.querySelector('.decrypted-content')
  if (decryptedContent) {
    decryptedContent.remove()
  }

  const promptOverlay = article.querySelector<HTMLElement>('.password-prompt-overlay')
  if (promptOverlay) {
    promptOverlay.style.display = 'flex'
  }

  const input = article.querySelector('.password-input') as HTMLInputElement
  if (input) {
    input.value = ''
  }

  clearUnlockTimer(slug)
  window.localStorage.removeItem(unlockKey(slug))
}

const scheduleReLock = (article: Element, slug: string, unlockUntil: number): void => {
  clearUnlockTimer(slug)
  const delay = Math.max(unlockUntil - Date.now(), 0)
  const timerId = window.setTimeout(() => {
    reLockContent(article, slug)
  }, delay)
  unlockTimers.set(slug, timerId)
}

document.addEventListener('nav', () => {
  const isProtected = window.document.body.dataset.protected === 'true'
  if (!isProtected) return

  const protectedArticles = document.querySelectorAll('[data-protected="true"]')

  protectedArticles.forEach(article => {
    const slug = article.getAttribute('data-slug')

    if (slug) {
      const unlockUntil = readUnlockUntil(slug)
      if (unlockUntil && unlockUntil > Date.now()) {
        scheduleReLock(article, slug, unlockUntil)
      } else {
        window.localStorage.removeItem(unlockKey(slug))
        clearUnlockTimer(slug)
      }
    }

    if (article.getAttribute('data-setup-complete') === 'true') {
      return
    }

    article.setAttribute('data-setup-complete', 'true')
    const form = article.querySelector('.password-form') as HTMLFormElement
    const input = article.querySelector('.password-input') as HTMLInputElement
    const errorEl = article.querySelector('.password-error') as HTMLElement
    const encryptedDataAttr = article.getAttribute('data-encrypted-content')

    if (!form || !input || !encryptedDataAttr || !slug) {
      article.removeAttribute('data-setup-complete')
      return
    }

    let encryptedData: EncryptedPayload
    try {
      const decoded = decodeURIComponent(encryptedDataAttr)
      encryptedData = JSON.parse(decoded)

      if (
        !encryptedData ||
        typeof encryptedData !== 'object' ||
        !encryptedData.salt ||
        typeof encryptedData.salt !== 'string' ||
        !encryptedData.iv ||
        typeof encryptedData.iv !== 'string' ||
        !encryptedData.ciphertext ||
        typeof encryptedData.ciphertext !== 'string'
      ) {
        console.error('invalid encrypted data structure:', encryptedData)
        return
      }
    } catch (err) {
      console.error('failed to parse encrypted data:', err)
      return
    }

    const isDecrypted = () => article.querySelector('.decrypted-content') !== null

    form.addEventListener('submit', async e => {
      e.preventDefault()
      const password = input.value.trim()
      if (!password) return

      if (isDecrypted()) {
        return
      }

      try {
        errorEl.style.display = 'none'

        const decryptedHtml = await decryptContent(encryptedData, password)

        const promptOverlay = article.querySelector<HTMLDivElement>('.password-prompt-overlay')
        if (promptOverlay) promptOverlay.style.display = 'none'

        const contentDiv = document.createElement('div')
        contentDiv.className = 'decrypted-content popover-hint'
        contentDiv.innerHTML = decryptedHtml
        article.appendChild(contentDiv)

        if (slug) {
          const unlockUntil = Date.now() + DECRYPTION_TTL
          window.localStorage.setItem(unlockKey(slug), unlockUntil.toString())
          scheduleReLock(article, slug, unlockUntil)
        }

        input.value = ''

        document.dispatchEvent(
          new CustomEvent('contentdecrypted', { detail: { article, content: contentDiv } }),
        )
      } catch (err) {
        console.error('decryption error:', err)
        errorEl.style.display = 'block'
        input.value = ''
        input.focus()
      }
    })
  })
})
