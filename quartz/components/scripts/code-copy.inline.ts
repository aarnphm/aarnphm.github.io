import { currentNavSignal } from './nav-lifecycle'

const INLINE_CODE_SELECTOR = 'code:not(pre > code):not(.no-copy)'
const hydratedCodeElements = new WeakSet<HTMLElement>()
let delegatedCodeCopySignal: AbortSignal | undefined

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (err) {
    console.error('Failed to copy:', err)
    return false
  }
}

function showCopyToast() {
  const event: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: { message: 'copied to clipboard', containerId: 'code-copy-toast-container' },
  })
  document.dispatchEvent(event)
}

function setupCodeCopyDelegation() {
  const signal = currentNavSignal()
  if (delegatedCodeCopySignal === signal) return
  delegatedCodeCopySignal = signal

  document.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const code = event.target.closest<HTMLElement>(INLINE_CODE_SELECTOR)
      if (!code || code.closest('.speech-play')) return
      event.preventDefault()
      const text = code.textContent || ''
      void copyToClipboard(text).then(success => {
        if (success && !signal.aborted) {
          showCopyToast()
        }
      })
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      if (delegatedCodeCopySignal === signal) delegatedCodeCopySignal = undefined
    },
    { once: true },
  )
}

function setupCodeCopy(root: Document | HTMLElement = document) {
  setupCodeCopyDelegation()
  const codeElements = root.querySelectorAll<HTMLElement>(INLINE_CODE_SELECTOR)
  codeElements.forEach(code => {
    if (code.closest('.speech-play')) return
    if (hydratedCodeElements.has(code)) return
    hydratedCodeElements.add(code)
    code.style.cursor = 'var(--cursor-pointer, pointer)'
    code.title = 'Click to copy'
  })
}

document.addEventListener('nav', () => {
  setupCodeCopy()
})
document.addEventListener('contentdecrypted', event => {
  setupCodeCopy(event.detail.content)
})
