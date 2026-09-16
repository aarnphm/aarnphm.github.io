import {
  arenaEmbedCapabilityPath,
  arenaEmbedCapturePath,
  arenaEmbedHtmlPath,
} from '../../util/arena-embed'
import { isRecord, readString } from '../../util/type-guards'

type EmbedMode = 'iframe' | 'fetch' | 'capture' | 'disabled'
type EmbedCapability = { mode: EmbedMode; finalUrl?: string }

function readCapability(value: unknown): EmbedCapability | null {
  if (!isRecord(value)) return null
  const mode = readString(value, 'mode')
  if (mode !== 'iframe' && mode !== 'fetch' && mode !== 'capture' && mode !== 'disabled') {
    return null
  }
  return { mode, finalUrl: readString(value, 'finalUrl') }
}

export function createArenaExternalEmbeds() {
  const capabilities = new Map<string, EmbedCapability>()
  const active = new Map<HTMLElement, AbortController>()

  function stop(host: HTMLElement) {
    active.get(host)?.abort()
    active.delete(host)
  }

  async function mountHost(host: HTMLElement) {
    const targetUrl = host.dataset.arenaUrl
    if (!host.isConnected || !targetUrl || active.has(host)) return
    const controller = new AbortController()
    const { signal } = controller
    active.set(host, controller)
    host.dataset.arenaEmbedStatus = 'loading'
    host.setAttribute('aria-busy', 'true')
    const loading = document.createElement('span')
    loading.className = 'arena-embed-loading'
    loading.setAttribute('role', 'status')
    loading.textContent = 'loading preview'
    host.replaceChildren(loading)

    const fail = (retry: boolean) => {
      if (signal.aborted || !host.isConnected) return
      stop(host)
      host.dataset.arenaEmbedStatus = 'error'
      host.removeAttribute('aria-busy')
      const shell = document.createElement('div')
      shell.className = 'arena-iframe-error'
      const content = document.createElement('div')
      content.className = 'arena-iframe-error-content'
      const message = document.createElement('p')
      message.textContent = 'preview unavailable'
      const link = document.createElement('a')
      link.href = targetUrl
      link.target = '_blank'
      link.rel = 'noopener noreferrer'
      link.className = 'arena-iframe-error-link'
      link.textContent = 'open original ↗'
      content.append(message, link)
      if (retry) {
        capabilities.delete(targetUrl)
        const button = document.createElement('button')
        button.type = 'button'
        button.textContent = 'retry'
        button.addEventListener('click', () => void mountHost(host), { once: true })
        content.append(button)
      }
      shell.append(content)
      host.replaceChildren(shell)
    }

    let capability: EmbedCapability | null
    const mode = host.dataset.arenaEmbedMode
    if (mode === 'none') {
      fail(false)
      return
    } else if (mode === 'iframe' || mode === 'fetch' || mode === 'capture') {
      capability = { mode }
    } else {
      capability = capabilities.get(targetUrl) ?? null
      if (!capability) {
        try {
          const response = await fetch(arenaEmbedCapabilityPath(targetUrl), {
            credentials: 'same-origin',
            cache: 'no-cache',
            signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
          })
          if (response.ok) capability = readCapability(await response.json())
        } catch {
          // An aborted modal must not update the next item's preview.
        }
        if (signal.aborted || !host.isConnected) return
        if (capability && capability.mode !== 'disabled') capabilities.set(targetUrl, capability)
      }
    }

    if (!capability || capability.mode === 'disabled') {
      fail(true)
      return
    }

    const timeout = window.setTimeout(() => fail(true), 60_000)
    signal.addEventListener('abort', () => window.clearTimeout(timeout), { once: true })
    const loaded = () => {
      if (signal.aborted) return
      window.clearTimeout(timeout)
      loading.remove()
      host.dataset.arenaEmbedStatus = 'loaded'
      host.removeAttribute('aria-busy')
    }
    const sourceUrl = capability.finalUrl ?? targetUrl
    if (capability.mode === 'capture') {
      const link = document.createElement('a')
      link.href = targetUrl
      link.target = '_blank'
      link.rel = 'noopener noreferrer'
      link.className = 'arena-modal-capture-link'
      const image = document.createElement('img')
      image.className = 'arena-modal-capture'
      image.dataset.ignorePopup = 'true'
      image.alt = `Captured preview: ${host.dataset.arenaTitle ?? targetUrl}`
      image.addEventListener('load', loaded, { once: true, signal })
      image.addEventListener('error', () => fail(true), { once: true, signal })
      const rect = host.getBoundingClientRect()
      image.src = arenaEmbedCapturePath(sourceUrl, {
        width: Math.round(rect.width || window.innerWidth),
        height: Math.round(Math.min(rect.height || window.innerHeight, window.innerHeight)),
        dpr: Math.min(2, Math.max(1, Math.ceil(window.devicePixelRatio || 1))),
      })
      link.append(image)
      host.append(link)
    } else {
      const fetched = capability.mode === 'fetch'
      const iframe = document.createElement('iframe')
      iframe.className = `arena-modal-iframe${fetched ? ' arena-modal-iframe-fetched' : ''}`
      iframe.title = `Embedded block: ${host.dataset.arenaTitle ?? targetUrl}`
      iframe.loading = 'eager'
      iframe.setAttribute(
        'sandbox',
        fetched
          ? ''
          : 'allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms',
      )
      if (fetched) iframe.referrerPolicy = 'no-referrer'
      iframe.addEventListener('load', loaded, { once: true, signal })
      iframe.src = fetched ? arenaEmbedHtmlPath(sourceUrl) : sourceUrl
      host.append(iframe)
    }
  }

  return {
    mount(root: HTMLElement) {
      root.querySelectorAll<HTMLElement>('.arena-modal-external-host').forEach(host => {
        void mountHost(host)
      })
    },
    cleanup() {
      for (const host of active.keys()) stop(host)
    },
  }
}
