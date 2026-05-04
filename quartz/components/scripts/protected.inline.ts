interface EncryptedPayload {
  ciphertext: string
  salt: string
  iv: string
}

const unlockTimers = new Map<string, number>()
type ProtectedContentContainer = Document | DocumentFragment | Element

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

const DECRYPTION_TTL = 30 * 60 * 1000

function protectedArticlesIn(container: ProtectedContentContainer): Element[] {
  return [
    ...(container instanceof Element && container.matches('[data-protected="true"]')
      ? [container]
      : []),
    ...Array.from(container.querySelectorAll('[data-protected="true"]')),
  ]
}

function isEncryptedPayload(value: unknown): value is EncryptedPayload {
  return (
    typeof value === 'object' &&
    value !== null &&
    'salt' in value &&
    typeof value.salt === 'string' &&
    'iv' in value &&
    typeof value.iv === 'string' &&
    'ciphertext' in value &&
    typeof value.ciphertext === 'string'
  )
}

function parseEncryptedPayload(article: Element): EncryptedPayload | null {
  const encryptedDataAttr = article.getAttribute('data-encrypted-content')
  if (!encryptedDataAttr) return null

  try {
    const decoded = decodeURIComponent(encryptedDataAttr)
    const encryptedData: unknown = JSON.parse(decoded)
    if (!isEncryptedPayload(encryptedData)) {
      console.error('invalid encrypted data structure:', encryptedData)
      return null
    }
    return encryptedData
  } catch (err) {
    console.error('failed to parse encrypted data:', err)
    return null
  }
}

function isDecrypted(article: Element): boolean {
  return article.querySelector('.decrypted-content') !== null
}

function isActiveProtectedArticle(article: Element): boolean {
  const entry = article.closest('.stream-entry')
  return !(entry instanceof HTMLElement && entry.hidden)
}

function lockedProtectedArticlesIn(container: ProtectedContentContainer): Element[] {
  return protectedArticlesIn(container).filter(
    article =>
      article.hasAttribute('data-encrypted-content') &&
      !isDecrypted(article) &&
      isActiveProtectedArticle(article),
  )
}

function primaryStream(container: ProtectedContentContainer = document): Element | Document {
  if (container instanceof Element) {
    return container.closest('.stream') ?? document.querySelector('.stream') ?? document
  }

  return document.querySelector('.stream') ?? document
}

function triggerHasLockedContent(trigger?: Element): boolean {
  const article = trigger?.closest('[data-protected="true"]')
  if (article) {
    return article.hasAttribute('data-encrypted-content') && !isDecrypted(article)
  }

  return lockedProtectedArticlesIn(primaryStream(trigger ?? document)).length > 0
}

function updateUnlockTriggers(container: ProtectedContentContainer = document): void {
  const stream = primaryStream(container)
  const hasLockedStreamArticles = lockedProtectedArticlesIn(stream).length > 0

  document.querySelectorAll<HTMLElement>('[data-protected-unlock-trigger]').forEach(trigger => {
    const article = trigger.closest('[data-protected="true"]')
    trigger.hidden = article
      ? !article.hasAttribute('data-encrypted-content') ||
        isDecrypted(article) ||
        !isActiveProtectedArticle(article)
      : !hasLockedStreamArticles
  })

  const panel =
    stream instanceof Element
      ? (stream.querySelector<HTMLElement>('[data-protected-unlock-panel]') ??
        document.querySelector<HTMLElement>('[data-protected-unlock-panel]'))
      : document.querySelector<HTMLElement>('[data-protected-unlock-panel]')

  if (!hasLockedStreamArticles && panel) {
    panel.hidden = true
    panel.classList.remove('is-open')
  }
}

function reLockContent(article: Element, slug: string): void {
  const decryptedContent = article.querySelector('.decrypted-content')
  if (decryptedContent) {
    decryptedContent.remove()
  }

  const promptOverlay = article.querySelector<HTMLElement>('.password-prompt-overlay')
  if (promptOverlay) {
    promptOverlay.style.display = 'flex'
  }

  article.querySelectorAll<HTMLElement>('[data-protected-unlock-trigger]').forEach(trigger => {
    trigger.hidden = false
  })

  const input = article.querySelector<HTMLInputElement>('.password-input')
  if (input) {
    input.value = ''
  }

  clearUnlockTimer(slug)
  window.localStorage.removeItem(unlockKey(slug))
  updateUnlockTriggers(article)
}

const scheduleReLock = (article: Element, slug: string, unlockUntil: number): void => {
  clearUnlockTimer(slug)
  const delay = Math.max(unlockUntil - Date.now(), 0)
  const timerId = window.setTimeout(() => {
    reLockContent(article, slug)
  }, delay)
  unlockTimers.set(slug, timerId)
}

async function unlockArticle(article: Element, password: string): Promise<HTMLElement> {
  const existingContent = article.querySelector<HTMLElement>('.decrypted-content')
  if (existingContent) {
    updateUnlockTriggers(article)
    return existingContent
  }

  const slug = article.getAttribute('data-slug')
  const encryptedData = parseEncryptedPayload(article)
  if (!slug || !encryptedData) throw new Error('missing protected content payload')

  const decryptedHtml = await decryptContent(encryptedData, password)

  const promptOverlay = article.querySelector<HTMLElement>('.password-prompt-overlay')
  if (promptOverlay) promptOverlay.style.display = 'none'

  article.querySelectorAll<HTMLElement>('[data-protected-unlock-trigger]').forEach(trigger => {
    trigger.hidden = true
  })

  const contentDiv = document.createElement('div')
  contentDiv.className = 'decrypted-content popover-hint'
  contentDiv.innerHTML = decryptedHtml
  article.appendChild(contentDiv)

  const unlockUntil = Date.now() + DECRYPTION_TTL
  window.localStorage.setItem(unlockKey(slug), unlockUntil.toString())
  scheduleReLock(article, slug, unlockUntil)
  updateUnlockTriggers(article)

  document.dispatchEvent(
    new CustomEvent('contentdecrypted', { detail: { article, content: contentDiv } }),
  )

  return contentDiv
}

function findUnlockPanel(trigger?: Element): HTMLElement | null {
  const stream = trigger?.closest('.stream') ?? document.querySelector('.stream')
  return (
    stream?.querySelector<HTMLElement>('[data-protected-unlock-panel]') ??
    document.querySelector<HTMLElement>('[data-protected-unlock-panel]')
  )
}

function showUnlockPanel(trigger?: Element): void {
  if (!triggerHasLockedContent(trigger)) {
    updateUnlockTriggers(trigger ?? document)
    return
  }

  const panel = findUnlockPanel(trigger)
  if (panel) {
    panel.hidden = false
    panel.classList.add('is-open')
    window.requestAnimationFrame(() => {
      panel.querySelector<HTMLInputElement>('.password-input')?.focus()
    })
    return
  }

  const article = trigger?.closest('[data-protected="true"]')
  const input =
    article?.querySelector<HTMLInputElement>('.password-input') ??
    document.querySelector<HTMLInputElement>('.password-input')
  input?.focus()
}

function setupUnlockTriggers(container: ProtectedContentContainer): void {
  const triggers = [
    ...(container instanceof Element && container.matches('[data-protected-unlock-trigger]')
      ? [container]
      : []),
    ...Array.from(container.querySelectorAll('[data-protected-unlock-trigger]')),
  ].filter((trigger): trigger is HTMLElement => trigger instanceof HTMLElement)

  triggers.forEach(trigger => {
    if (trigger.dataset.protectedUnlockTriggerSetup === 'true') return
    trigger.dataset.protectedUnlockTriggerSetup = 'true'

    trigger.addEventListener('click', event => {
      event.preventDefault()
      showUnlockPanel(trigger)
    })
  })
}

function setupUnlockAllForms(container: ProtectedContentContainer): void {
  const forms = [
    ...(container instanceof Element && container.matches('[data-protected-unlock-all="true"]')
      ? [container]
      : []),
    ...Array.from(container.querySelectorAll('[data-protected-unlock-all="true"]')),
  ].filter((form): form is HTMLFormElement => form instanceof HTMLFormElement)

  forms.forEach(form => {
    if (form.dataset.protectedUnlockAllSetup === 'true') return
    form.dataset.protectedUnlockAllSetup = 'true'

    form.addEventListener('submit', async event => {
      event.preventDefault()

      const input = form.querySelector<HTMLInputElement>('.password-input')
      if (!input) return

      const password = input.value.trim()
      if (!password) return

      const panel = form.closest<HTMLElement>('[data-protected-unlock-panel]')
      const errorEl =
        panel?.querySelector<HTMLElement>('.password-error') ??
        form.querySelector<HTMLElement>('.password-error')
      const stream = form.closest('.stream') ?? document.querySelector('.stream') ?? document
      const articles = lockedProtectedArticlesIn(stream)

      if (articles.length === 0) {
        input.value = ''
        if (panel) {
          panel.hidden = true
          panel.classList.remove('is-open')
        }
        updateUnlockTriggers(stream)
        return
      }

      try {
        if (errorEl) errorEl.style.display = 'none'
        await Promise.all(articles.map(article => unlockArticle(article, password)))
        input.value = ''
        if (panel) {
          panel.hidden = true
          panel.classList.remove('is-open')
        }
        updateUnlockTriggers(stream)
      } catch (err) {
        console.error('decryption error:', err)
        if (errorEl) errorEl.style.display = 'block'
        input.value = ''
        input.focus()
      }
    })
  })
}

function setupProtectedContent(container: ProtectedContentContainer = document): void {
  setupUnlockTriggers(container)
  setupUnlockAllForms(container)

  const protectedArticles = protectedArticlesIn(container)

  if (protectedArticles.length === 0) {
    updateUnlockTriggers(container)
    return
  }

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
    const form = article.querySelector<HTMLFormElement>('.password-form')
    const input = article.querySelector<HTMLInputElement>('.password-input')
    const errorEl = article.querySelector<HTMLElement>('.password-error')
    const encryptedDataAttr = article.getAttribute('data-encrypted-content')

    if (!encryptedDataAttr || !slug) {
      article.removeAttribute('data-setup-complete')
      return
    }

    if (!form || !input) {
      return
    }

    form.addEventListener('submit', async e => {
      e.preventDefault()
      const password = input.value.trim()
      if (!password) return

      if (isDecrypted(article)) {
        return
      }

      try {
        if (errorEl) errorEl.style.display = 'none'
        await unlockArticle(article, password)
        input.value = ''
      } catch (err) {
        console.error('decryption error:', err)
        if (errorEl) errorEl.style.display = 'block'
        input.value = ''
        input.focus()
      }
    })
  })

  updateUnlockTriggers(container)
}

document.addEventListener('nav', () => {
  setupProtectedContent()
})

document.addEventListener('protectedcontentloaded', event => {
  const detail = event instanceof CustomEvent ? event.detail : null
  const container =
    detail &&
    typeof detail === 'object' &&
    'container' in detail &&
    (detail.container instanceof Document ||
      detail.container instanceof DocumentFragment ||
      detail.container instanceof Element)
      ? detail.container
      : document

  setupProtectedContent(container)
})
