interface EncryptedPayload {
  ciphertext: string
  salt: string
  iv: string
}

async function deriveKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const encoder = new TextEncoder()
  const passwordKey = await crypto.subtle.importKey(
    "raw",
    encoder.encode(password),
    "PBKDF2",
    false,
    ["deriveKey"],
  )

  return crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt,
      iterations: 100000,
      hash: "SHA-256",
    },
    passwordKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"],
  )
}

async function decryptContent(encryptedData: EncryptedPayload, password: string): Promise<string> {
  // Convert base64url back to base64 (add padding and restore +/)
  const fromBase64Url = (str: string | undefined): string => {
    if (!str || typeof str !== "string") {
      throw new Error(`invalid base64url string: ${str}`)
    }
    // Replace URL-safe characters back to standard base64
    let base64 = str.replace(/-/g, "+").replace(/_/g, "/")
    // Add padding
    while (base64.length % 4) {
      base64 += "="
    }
    return base64
  }

  const salt = Uint8Array.from(atob(fromBase64Url(encryptedData.salt)), (c) => c.charCodeAt(0))
  const iv = Uint8Array.from(atob(fromBase64Url(encryptedData.iv)), (c) => c.charCodeAt(0))
  const ciphertext = Uint8Array.from(atob(fromBase64Url(encryptedData.ciphertext)), (c) =>
    c.charCodeAt(0),
  )

  const key = await deriveKey(password, salt)

  try {
    const decrypted = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ciphertext)

    return new TextDecoder().decode(decrypted)
  } catch {
    throw new Error("decryption failed")
  }
}

// TTL for decrypted content in milliseconds (30 minutes)
const DECRYPTION_TTL = 30 * 60 * 1000

// Immediately hide overlay for pages with valid stored passwords to prevent flash
;(() => {
  const protectedArticles = document.querySelectorAll('[data-protected="true"]')
  protectedArticles.forEach((article) => {
    const slug = article.getAttribute("data-slug")
    if (!slug) return

    const storedPassword = sessionStorage.getItem(`password:${slug}`)
    const decryptedAtStr = sessionStorage.getItem(`decrypted-at:${slug}`)

    if (storedPassword && decryptedAtStr) {
      const decryptedAt = parseInt(decryptedAtStr, 10)
      const elapsed = Date.now() - decryptedAt

      // If password exists and TTL hasn't expired, hide overlay immediately
      if (elapsed < DECRYPTION_TTL) {
        const promptOverlay = article.querySelector<HTMLElement>(".password-prompt-overlay")
        if (promptOverlay) {
          promptOverlay.style.display = "none"
        }
      }
    }
  })
})()

document.addEventListener("nav", () => {
  const protectedArticles = document.querySelectorAll('[data-protected="true"]')

  // Function to re-lock content after TTL
  const reLockContent = (article: Element, slug: string) => {
    const decryptedContent = article.querySelector(".decrypted-content")
    const promptOverlay = article.querySelector<HTMLElement>(".password-prompt-overlay")

    // Remove decrypted content
    if (decryptedContent) {
      decryptedContent.remove()
    }

    // Show prompt overlay
    if (promptOverlay) {
      promptOverlay.style.display = "flex"
    }

    // Clear stored password and timestamp
    sessionStorage.removeItem(`password:${slug}`)
    sessionStorage.removeItem(`decrypted-at:${slug}`)

    // Clear input and setup flag
    const input = article.querySelector(".password-input") as HTMLInputElement
    if (input) {
      input.value = ""
    }
    article.removeAttribute("data-setup-complete")
  }

  protectedArticles.forEach((article) => {
    // Skip if already set up (prevents duplicate event listeners)
    if (article.getAttribute("data-setup-complete") === "true") {
      return
    }

    // Mark as set up
    article.setAttribute("data-setup-complete", "true")
    const form = article.querySelector(".password-form") as HTMLFormElement
    const input = article.querySelector(".password-input") as HTMLInputElement
    const errorEl = article.querySelector(".password-error") as HTMLElement
    const encryptedDataAttr = article.getAttribute("data-encrypted-content")
    const slug = article.getAttribute("data-slug")

    if (!form || !input || !encryptedDataAttr) return

    let encryptedData: EncryptedPayload
    try {
      const decoded = decodeURIComponent(encryptedDataAttr)
      encryptedData = JSON.parse(decoded)

      // Validate encrypted data structure
      if (
        !encryptedData ||
        typeof encryptedData !== "object" ||
        !encryptedData.salt ||
        typeof encryptedData.salt !== "string" ||
        !encryptedData.iv ||
        typeof encryptedData.iv !== "string" ||
        !encryptedData.ciphertext ||
        typeof encryptedData.ciphertext !== "string"
      ) {
        console.error("invalid encrypted data structure:", encryptedData)
        return
      }
    } catch (err) {
      console.error("failed to parse encrypted data:", err)
      return
    }

    // Check if content is already decrypted (singleton check)
    const isDecrypted = () => article.querySelector(".decrypted-content") !== null

    form.addEventListener("submit", async (e) => {
      e.preventDefault()
      const password = input.value.trim()

      if (!password) return

      // If already decrypted, do nothing (singleton)
      if (isDecrypted()) {
        return
      }

      try {
        errorEl.style.display = "none"

        const decryptedHtml = await decryptContent(encryptedData, password)

        // Hide the password prompt overlay
        const promptOverlay = article.querySelector<HTMLDivElement>(".password-prompt-overlay")
        if (promptOverlay) promptOverlay.style.display = "none"

        // Create and insert decrypted content with fade-in
        const contentDiv = document.createElement("div")
        contentDiv.className = "decrypted-content popover-hint"
        contentDiv.innerHTML = decryptedHtml
        article.appendChild(contentDiv)

        // Store password and decryption timestamp
        if (slug) {
          sessionStorage.setItem(`password:${slug}`, password)
          sessionStorage.setItem(`decrypted-at:${slug}`, Date.now().toString())

          // Set TTL timeout to re-lock
          setTimeout(() => {
            reLockContent(article, slug)
          }, DECRYPTION_TTL)
        }

        document.dispatchEvent(
          new CustomEvent("content-decrypted", {
            detail: { article, contentDiv },
          }),
        )
      } catch (err) {
        console.error("decryption error:", err)
        errorEl.style.display = "block"
        input.value = ""
        input.focus()
      }
    })

    // Auto-decrypt if password in session and within TTL
    // This runs only once during initial setup
    const storedPassword = sessionStorage.getItem(`password:${slug}`)
    const decryptedAtStr = sessionStorage.getItem(`decrypted-at:${slug}`)

    if (storedPassword && decryptedAtStr) {
      const decryptedAt = parseInt(decryptedAtStr, 10)
      const now = Date.now()
      const elapsed = now - decryptedAt

      // Check if TTL has expired
      if (elapsed < DECRYPTION_TTL) {
        if (form && input) {
          // Wait for DOM to be fully ready before auto-decrypting
          requestAnimationFrame(() => {
            input.value = storedPassword
            form.dispatchEvent(new Event("submit"))
          })

          // Set remaining TTL timeout
          const remainingTTL = DECRYPTION_TTL - elapsed
          setTimeout(() => {
            const articleEl = document.querySelector(`[data-slug="${slug}"]`)
            if (articleEl) {
              reLockContent(articleEl, slug!)
            }
          }, remainingTTL)
        }
      } else {
        // TTL expired, clear stored data
        sessionStorage.removeItem(`password:${slug}`)
        sessionStorage.removeItem(`decrypted-at:${slug}`)
      }
    }
  })
})
