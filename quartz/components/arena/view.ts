import type { ArenaEvent, SearchResultOptions, SearchScope } from './model'
import {
  arenaEmbedCapabilityPath,
  arenaEmbedCapturePath,
  type ArenaEmbedCaptureOptions,
  arenaEmbedHtmlPath,
  arenaPdfFilenameFromUrl,
  arenaPdfViewerSource,
  isArenaPdfUrl,
  type ArenaExternalEmbedMode,
} from '../../util/arena-embed'
import { normalizeRelativeURLs } from '../../util/path'
import { isRecord, readString } from '../../util/type-guards'
import { loadMapbox, applyMonochromeMapPalette } from '../scripts/mapbox-client'
import { fetchCanonical, tokenizeTerm } from '../scripts/util'

let currentBlockIndex = 0
let totalBlocks = 0
const p = new DOMParser()

const SUBSTACK_POST_REGEX = /^https?:\/\/[^/]+\/p\/[^/]+/i
const mapInstances = new WeakMap<HTMLElement, any>()
let scrollLockState: { x: number; y: number } | null = null
let searchInput: HTMLInputElement | null = null
let activeResultIndex: number | null = null
let registerCleanup: ((cleanup: () => void) => void) | null = null
const arenaEmbedCapabilityCache = new Map<string, Promise<ArenaEmbedCapability | null>>()

interface ArenaEmbedCapability {
  mode: 'iframe' | 'fetch' | 'capture' | 'disabled'
  finalUrl?: string
  reason?: string
}

const addCleanup = (cleanup: () => void) => {
  if (registerCleanup) {
    registerCleanup(cleanup)
  }
}

function readArenaEmbedCapability(value: unknown): ArenaEmbedCapability | null {
  if (!isRecord(value)) return null

  const mode = readString(value, 'mode')
  if (mode !== 'iframe' && mode !== 'fetch' && mode !== 'capture' && mode !== 'disabled') {
    return null
  }

  const finalUrl = readString(value, 'finalUrl')
  const reason = readString(value, 'reason')
  const capability: ArenaEmbedCapability = { mode }
  if (finalUrl) capability.finalUrl = finalUrl
  if (reason) capability.reason = reason
  return capability
}

function readArenaExternalModeDataset(value: string | undefined): ArenaExternalEmbedMode {
  if (value === 'iframe' || value === 'fetch' || value === 'capture' || value === 'none') {
    return value
  }
  return 'auto'
}

async function fetchArenaEmbedCapability(rawUrl: string): Promise<ArenaEmbedCapability | null> {
  const cached = arenaEmbedCapabilityCache.get(rawUrl)
  if (cached) return cached

  const pending = fetch(arenaEmbedCapabilityPath(rawUrl), { credentials: 'same-origin' })
    .then(async response => {
      if (!response.ok) return null
      return readArenaEmbedCapability(await response.json())
    })
    .catch(error => {
      console.error(error)
      return null
    })

  arenaEmbedCapabilityCache.set(rawUrl, pending)
  return pending
}

function renderExternalEmbedFallback(host: HTMLElement, targetUrl: string) {
  host.innerHTML = ''
  const shell = document.createElement('div')
  shell.className = 'arena-iframe-error'
  const content = document.createElement('div')
  content.className = 'arena-iframe-error-content'
  const message = document.createElement('p')
  message.textContent = 'embedded content unavailable'
  const link = document.createElement('a')
  link.href = targetUrl
  link.target = '_blank'
  link.rel = 'noopener noreferrer'
  link.className = 'arena-iframe-error-link'
  link.textContent = 'open in new tab ->'
  content.append(message, link)
  shell.appendChild(content)
  host.appendChild(shell)
}

function renderFetchedExternalEmbed(host: HTMLElement, targetUrl: string) {
  let iframe = host.querySelector<HTMLIFrameElement>('iframe.arena-modal-iframe')
  if (!iframe) {
    host.innerHTML = ''
    iframe = document.createElement('iframe')
    iframe.className = 'arena-modal-iframe'
    iframe.loading = 'lazy'
    const blockId = host.dataset.blockId
    if (blockId) iframe.dataset.blockId = blockId
    host.appendChild(iframe)
  }

  iframe.classList.add('arena-modal-iframe-fetched')
  iframe.setAttribute('sandbox', '')
  iframe.referrerPolicy = 'no-referrer'
  iframe.src = arenaEmbedHtmlPath(targetUrl)
}

function captureOptionsForHost(host: HTMLElement): ArenaEmbedCaptureOptions {
  const rect = host.getBoundingClientRect()
  const width = Math.round(rect.width || host.clientWidth || window.innerWidth)
  const hostHeight = rect.height || host.clientHeight || window.innerHeight
  const height = Math.round(Math.min(hostHeight, window.innerHeight))
  const dpr = Math.min(2, Math.max(1, Math.ceil(window.devicePixelRatio || 1)))
  return { width, height, dpr }
}

function renderCapturedExternalEmbed(host: HTMLElement, targetUrl: string) {
  host.innerHTML = ''
  const link = document.createElement('a')
  link.href = targetUrl
  link.target = '_blank'
  link.rel = 'noopener noreferrer'
  link.className = 'arena-modal-capture-link'
  const image = document.createElement('img')
  image.className = 'arena-modal-capture'
  image.loading = 'lazy'
  image.decoding = 'async'
  image.alt = 'Captured preview'
  const captureOptions = captureOptionsForHost(host)
  image.width = captureOptions.width ?? 0
  image.height = captureOptions.height ?? 0
  image.src = arenaEmbedCapturePath(targetUrl, captureOptions)
  link.appendChild(image)
  host.appendChild(link)
}

async function hydrateExternalEmbedHost(host: HTMLElement) {
  if (!host.isConnected) return
  const targetUrl = host.dataset.arenaUrl
  if (!targetUrl) return

  const mode = readArenaExternalModeDataset(host.dataset.arenaEmbedMode)
  if (host.dataset.arenaEmbedStatus === 'loading' || host.dataset.arenaEmbedStatus === 'loaded') {
    return
  }

  if (mode === 'none') {
    renderExternalEmbedFallback(host, targetUrl)
    host.dataset.arenaEmbedStatus = 'loaded'
    return
  }

  if (mode === 'fetch') {
    renderFetchedExternalEmbed(host, targetUrl)
    host.dataset.arenaEmbedStatus = 'loaded'
    return
  }

  if (mode === 'capture') {
    renderCapturedExternalEmbed(host, targetUrl)
    host.dataset.arenaEmbedStatus = 'loaded'
    return
  }

  if (mode === 'iframe') {
    host.dataset.arenaEmbedStatus = 'loaded'
    return
  }

  host.dataset.arenaEmbedStatus = 'loading'
  const capability = await fetchArenaEmbedCapability(targetUrl)
  if (!host.isConnected) return
  if (!capability) {
    host.dataset.arenaEmbedStatus = 'loaded'
    return
  }

  if (capability.mode === 'fetch') {
    renderFetchedExternalEmbed(host, capability.finalUrl ?? targetUrl)
  } else if (capability.mode === 'capture') {
    renderCapturedExternalEmbed(host, capability.finalUrl ?? targetUrl)
  } else if (capability.mode === 'disabled') {
    renderExternalEmbedFallback(host, capability.finalUrl ?? targetUrl)
  }

  host.dataset.arenaEmbedStatus = 'loaded'
}

function hydrateExternalEmbeds(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>('.arena-modal-external-host[data-arena-url]').forEach(host => {
    void hydrateExternalEmbedHost(host)
  })
}

function lockPageScroll() {
  if (scrollLockState) return
  scrollLockState = { x: window.scrollX, y: window.scrollY }
  document.documentElement.classList.add('arena-modal-open')
  document.body.classList.add('arena-modal-open')
  document.body.style.position = 'fixed'
  document.body.style.top = `-${scrollLockState.y}px`
  document.body.style.left = '0'
  document.body.style.right = '0'
  document.body.style.width = '100%'
  document.body.style.overflow = 'hidden'
}

function unlockPageScroll() {
  const state = scrollLockState
  document.documentElement.classList.remove('arena-modal-open')
  document.body.classList.remove('arena-modal-open')
  document.body.style.position = ''
  document.body.style.top = ''
  document.body.style.left = ''
  document.body.style.right = ''
  document.body.style.width = ''
  document.body.style.overflow = ''
  scrollLockState = null
  if (state) {
    window.scrollTo(state.x, state.y)
  }
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, char => {
    switch (char) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      case "'":
        return '&#39;'
      default:
        return char
    }
  })
}

function appendHighlightedText(parent: HTMLElement, searchTerm: string, text: string) {
  const terms = tokenizeTerm(searchTerm)
    .map(term => term.toLowerCase())
    .filter(term => term.length > 0)
    .sort((a, b) => b.length - a.length)
  if (terms.length === 0) {
    parent.textContent = text
    return
  }

  const lowerText = text.toLowerCase()
  let cursor = 0
  while (cursor < text.length) {
    let nextIndex = -1
    let nextTerm = ''
    for (const term of terms) {
      const index = lowerText.indexOf(term, cursor)
      if (index === -1) continue
      if (
        nextIndex === -1 ||
        index < nextIndex ||
        (index === nextIndex && term.length > nextTerm.length)
      ) {
        nextIndex = index
        nextTerm = term
      }
    }

    if (nextIndex === -1) {
      parent.append(document.createTextNode(text.slice(cursor)))
      return
    }
    if (nextIndex > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, nextIndex)))
    }
    const highlight = document.createElement('span')
    highlight.className = 'highlight'
    highlight.textContent = text.slice(nextIndex, nextIndex + nextTerm.length)
    parent.append(highlight)
    cursor = nextIndex + nextTerm.length
  }
}

function renderMapFallback(node: HTMLElement, message: string) {
  node.dataset.mapStatus = 'error'
  node.classList.add('arena-map-error')
  node.textContent = message
}

function cleanupMaps(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>('.arena-modal-map[data-map-initialized]').forEach(node => {
    const map = mapInstances.get(node)
    if (map) {
      try {
        map.remove()
      } catch (error) {
        console.error(error)
      }
      mapInstances.delete(node)
    }
    node.removeAttribute('data-map-initialized')
    node.removeAttribute('data-map-status')
    node.classList.remove('arena-map-error')
    node.textContent = ''
  })
}

function hydrateMapboxMaps(root: HTMLElement) {
  const mapNodes = root.querySelectorAll<HTMLElement>(
    '.arena-modal-map[data-map-lon][data-map-lat]',
  )
  if (mapNodes.length === 0) {
    return
  }

  mapNodes.forEach(node => {
    if (node.dataset.mapInitialized === '1') return
    node.dataset.mapStatus = 'loading'
    node.classList.remove('arena-map-error')
    node.textContent = 'loading map…'
  })

  loadMapbox()
    .then(mapboxgl => {
      if (!mapboxgl) {
        mapNodes.forEach(node => renderMapFallback(node, 'map unavailable'))
        return
      }

      mapNodes.forEach(node => {
        if (!node.isConnected || node.dataset.mapInitialized === '1') {
          return
        }

        const lon = Number.parseFloat(node.dataset.mapLon || '')
        const lat = Number.parseFloat(node.dataset.mapLat || '')

        if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
          renderMapFallback(node, 'invalid location')
          return
        }

        try {
          const map = new mapboxgl.Map({
            container: node,
            style: 'mapbox://styles/mapbox/light-v11',
            center: [lon, lat],
            zoom: 15,
            attributionControl: false,
          })

          mapInstances.set(node, map)
          node.dataset.mapInitialized = '1'
          node.dataset.mapStatus = 'loading'

          const markerEl = document.createElement('div')
          markerEl.className = 'base-map-marker'
          markerEl.textContent = '•'
          markerEl.style.color = '#2b2418'
          markerEl.style.width = '24px'
          markerEl.style.height = '24px'
          markerEl.style.display = 'flex'
          markerEl.style.alignItems = 'center'
          markerEl.style.justifyContent = 'center'
          markerEl.style.fontSize = '18px'
          markerEl.style.cursor = 'pointer'

          const marker = new mapboxgl.Marker({ element: markerEl, anchor: 'bottom' }).setLngLat([
            lon,
            lat,
          ])
          const title = node.dataset.mapTitle
          if (title && title.trim().length > 0) {
            const popup = new mapboxgl.Popup({
              closeButton: false,
              closeOnClick: false,
              offset: 25,
            })
            popup.setText(title)
            marker.setPopup(popup)
          }
          marker.addTo(map)

          map.once('load', () => {
            applyMonochromeMapPalette(map)
            node.dataset.mapStatus = 'loaded'
            node.classList.remove('arena-map-error')
            try {
              map.resize()
            } catch (error) {
              console.error(error)
            }
            if (title && marker.getPopup()) {
              try {
                marker.togglePopup()
              } catch (error) {
                console.error(error)
              }
            }
          })

          map.on('error', () => {
            if (node.dataset.mapStatus !== 'loaded') {
              renderMapFallback(node, 'map unavailable')
            }
          })
        } catch (error) {
          console.error(error)
          renderMapFallback(node, 'map unavailable')
        }
      })
    })
    .catch(error => {
      console.error(error)
      mapNodes.forEach(node => renderMapFallback(node, 'map unavailable'))
    })
}

function injectSubstackScript(container: HTMLElement) {
  const script = document.createElement('script')
  script.async = true
  script.src = 'https://substack.com/embedjs/embed.js'
  container.appendChild(script)
}

function hydrateSubstackEmbeds(root: HTMLElement) {
  const nodes = root.querySelectorAll<HTMLElement>(
    '.arena-modal-embed-substack[data-substack-url]:not([data-substack-status])',
  )
  nodes.forEach(node => {
    const targetUrl = node.dataset.substackUrl
    if (!targetUrl) return
    node.dataset.substackStatus = 'loading'
    node.classList.add('arena-modal-embed')
    node.innerHTML = ''
    const wrapper = document.createElement('div')
    wrapper.className = 'substack-post-embed'
    const link = document.createElement('a')
    link.href = targetUrl
    link.textContent = 'Read on Substack'
    link.setAttribute('data-post-link', '')
    wrapper.appendChild(link)
    node.appendChild(wrapper)
    injectSubstackScript(node)
    node.dataset.substackStatus = 'loaded'
  })
}

function ensureInternalPreviewContainer(host: HTMLElement): HTMLDivElement {
  let preview = host.querySelector('.arena-modal-internal-preview') as HTMLDivElement | null
  if (!preview) {
    preview = document.createElement('div')
    preview.className = 'arena-modal-internal-preview'
    host.appendChild(preview)
  }
  return preview
}

function renderInternalPreviewLoading(container: HTMLElement) {
  container.innerHTML = ''
  const spinner = document.createElement('span')
  spinner.className = 'arena-loading-spinner'
  spinner.setAttribute('role', 'status')
  spinner.setAttribute('aria-label', 'Loading preview')
  container.appendChild(spinner)
}

function renderInternalPreviewError(container: HTMLElement) {
  container.innerHTML = ''
  const message = document.createElement('p')
  message.textContent = 'Unable to load preview.'
  container.appendChild(message)
}

async function hydrateInternalHost(host: HTMLElement) {
  if (!host.isConnected) return
  const href = host.dataset.internalHref
  if (
    !href ||
    host.dataset.internalStatus === 'loading' ||
    host.dataset.internalStatus === 'loaded'
  ) {
    return
  }

  host.dataset.internalStatus = 'loading'
  const preview = ensureInternalPreviewContainer(host)
  renderInternalPreviewLoading(preview)

  try {
    const targetUrl = new URL(href, window.location.origin)
    const urlWithoutHash = new URL(targetUrl.toString())
    urlWithoutHash.hash = ''

    const isPdf = /\.pdf(?:[?#].*)?$/i.test(urlWithoutHash.pathname)

    if (isPdf) {
      const filename = urlWithoutHash.pathname.split('/').pop() || 'document.pdf'
      preview.innerHTML = ''
      preview.classList.add('arena-modal-embed', 'arena-modal-embed-pdf', 'pdf-embed')
      preview.dataset.pdfSrc = `${urlWithoutHash.pathname}${urlWithoutHash.search}`
      preview.dataset.pdfTitle = filename
      preview.dataset.pdfFit = 'page'
      mountPdfEmbeds(preview)
      host.dataset.internalStatus = 'loaded'
      return
    }

    const response = await fetchCanonical(urlWithoutHash)
    if (!response.ok) {
      throw new Error(`status ${response.status}`)
    }

    const headerContentType = response.headers.get('Content-Type')
    const contentType = headerContentType?.split(';')[0]
    if (!contentType || !contentType.startsWith('text/html')) {
      throw new Error('non-html')
    }

    const contents = await response.text()
    const html = p.parseFromString(contents, 'text/html')
    normalizeRelativeURLs(html, targetUrl)
    html.querySelectorAll('[id]').forEach(el => {
      if (el.id && el.id.length > 0) {
        el.id = `arena-modal-${el.id}`
      }
    })

    const hints = [
      ...(html.getElementsByClassName('popover-hint') as HTMLCollectionOf<HTMLElement>),
    ]
    preview.innerHTML = ''

    if (hints.length === 0) {
      renderInternalPreviewError(preview)
      host.dataset.internalStatus = 'error'
      return
    }

    for (const hint of hints) {
      preview.appendChild(document.importNode(hint, true))
    }

    const hashValue = host.dataset.internalHash
    if (hashValue) {
      const normalized = hashValue.startsWith('#') ? hashValue.slice(1) : hashValue
      const targetId = `arena-modal-${normalized}`
      const anchorCandidates = preview.querySelectorAll<HTMLElement>('[id]')
      const anchor = Array.from(anchorCandidates).find(el => el.id === targetId) ?? null
      if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth' })
      }
    }

    host.dataset.internalStatus = 'loaded'
  } catch (error) {
    console.error(error)
    renderInternalPreviewError(preview)
    host.dataset.internalStatus = 'error'
  }
}

function hydrateInternalHosts(root: HTMLElement) {
  const hosts = root.querySelectorAll<HTMLElement>('.arena-modal-internal-host[data-internal-href]')
  hosts.forEach(host => {
    void hydrateInternalHost(host)
  })
}

function mountPdfEmbeds(root: HTMLElement) {
  window.quartzPdfEmbeds?.mount(root)
}

function cleanupPdfs(root: HTMLElement) {
  window.quartzPdfEmbeds?.cleanup(root)
}

function renderExternalModalHtml(
  block: ArenaBlockSearchable,
  targetUrl: string,
  mode: ArenaExternalEmbedMode,
): string {
  const escapedUrl = escapeHtml(targetUrl)
  if (mode === 'none') {
    return `
      <div class="arena-iframe-error">
        <div class="arena-iframe-error-content">
          <p>embedded content unavailable</p>
          <a href="${escapedUrl}" target="_blank" rel="noopener noreferrer" class="arena-iframe-error-link">
            open in new tab ->
          </a>
        </div>
      </div>
    `
  }

  if (mode === 'capture') {
    return `
    <div
      class="arena-modal-external-host"
      data-block-id="${escapeHtml(block.id)}"
      data-arena-url="${escapedUrl}"
      data-arena-embed-mode="${mode}"
    >
      <a href="${escapedUrl}" target="_blank" rel="noopener noreferrer" class="arena-modal-capture-link">
        <img
          class="arena-modal-capture"
          loading="lazy"
          decoding="async"
          alt="Captured preview: ${escapeHtml(block.title ?? block.content ?? 'Block')}"
          src="${escapeHtml(arenaEmbedCapturePath(targetUrl))}"
        />
      </a>
    </div>
  `
  }

  const fetched = mode === 'fetch'
  const frameTitle = escapeHtml(block.title ?? block.content ?? 'Block')
  const iframeSrc = fetched ? arenaEmbedHtmlPath(targetUrl) : targetUrl
  const sandbox = fetched
    ? 'sandbox="" referrerpolicy="no-referrer"'
    : 'sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms"'

  return `
    <div
      class="arena-modal-external-host"
      data-block-id="${escapeHtml(block.id)}"
      data-arena-url="${escapedUrl}"
      data-arena-embed-mode="${mode}"
    >
      <iframe
        class="arena-modal-iframe${fetched ? ' arena-modal-iframe-fetched' : ''}"
        title="Embedded block: ${frameTitle}"
        loading="lazy"
        data-block-id="${escapeHtml(block.id)}"
        ${sandbox}
        src="${escapeHtml(iframeSrc)}"
      ></iframe>
    </div>
  `
}

function renderPdfModalHtml(pdfUrl: string): string {
  return `
    <div
      class="arena-modal-embed arena-modal-embed-pdf pdf-embed"
      data-pdf-src="${escapeHtml(arenaPdfViewerSource(pdfUrl))}"
      data-pdf-title="${escapeHtml(arenaPdfFilenameFromUrl(pdfUrl))}"
      data-pdf-fit="page"
    >
      <span class="pdf-embed-loading">Loading PDF</span>
    </div>
  `
}

function renderArenaSearchSubItemHtml(block: ArenaBlockSearchable): string {
  if (block.blockHtml) return block.blockHtml
  if (block.titleHtml) return block.titleHtml
  return escapeHtml(block.title ?? block.content)
}

function createModalDataFromJson(block: ArenaBlockSearchable, channelSlug: string): HTMLElement {
  const container = document.createElement('div')
  container.className = 'arena-block-modal-data'
  container.id = `arena-modal-data-${block.id}`
  container.dataset.blockId = block.id
  container.dataset.channelSlug = channelSlug
  container.style.display = 'none'

  const displayUrl =
    block.url ??
    (block.internalSlug ? `${window.location.origin}/${block.internalSlug}` : undefined)
  const escapedDisplayUrl = displayUrl ? escapeHtml(displayUrl) : ''

  const metadataHtml: string[] = []

  if (block.metadata) {
    const consumedKeys = new Set(['accessed', 'accessed_date', 'date', 'tags', 'tag', 'coord'])
    const entries = Object.entries(block.metadata)
      .filter(([key, value]) => {
        if (typeof value !== 'string' || value.trim().length === 0) return false
        if (consumedKeys.has(key.toLowerCase())) return false
        return true
      })
      .sort(([a], [b]) => a.localeCompare(b))

    for (const [key, value] of entries) {
      const label = key.replace(/_/g, ' ')
      metadataHtml.push(`
        <div class="arena-meta-item">
          <span class="arena-meta-label">${escapeHtml(label)}</span>
          <em class="arena-meta-value">${escapeHtml(value)}</em>
        </div>
      `)
    }
  }

  if (block.tags && block.tags.length > 0) {
    const tagsLabel = block.tags.length === 1 ? 'tag' : 'tags'
    const tagsList = block.tags
      .map(tag => `<span class="tag-link">${escapeHtml(tag)}</span>`)
      .join('')
    metadataHtml.push(`
      <div class="arena-meta-item">
        <span class="arena-meta-label">${tagsLabel}</span>
        <em class="arena-meta-value">
          <span class="arena-meta-taglist">${tagsList}</span>
        </em>
      </div>
    `)
  }

  const subItems = block.subItems ?? []
  const subItemsHtml =
    subItems.length > 0
      ? `
    <div class="arena-modal-connections">
      <div class="arena-modal-connections-header">
        <span class="arena-modal-connections-title">notes</span>
        <span class="arena-modal-connections-count">${subItems.length}</span>
      </div>
      <ul class="arena-modal-connections-list">
        ${subItems
          .map(
            subItem => `
          <li>${renderArenaSearchSubItemHtml(subItem)}</li>
        `,
          )
          .join('')}
      </ul>
    </div>
  `
      : ''

  const mapTitle = block.title || block.content || block.url || ''
  const mapHtml = block.coordinates
    ? `
        <div class="arena-modal-map-wrapper">
          <div
            class="arena-modal-map"
            data-map-lon="${String(block.coordinates.lon)}"
            data-map-lat="${String(block.coordinates.lat)}"${mapTitle.trim().length > 0 ? ` data-map-title="${escapeHtml(mapTitle)}"` : ''}
          ></div>
        </div>
      `
    : ''

  let mainContentHtml = ''
  if (block.embedHtml) {
    mainContentHtml = block.embedHtml
  } else if (block.url && SUBSTACK_POST_REGEX.test(block.url)) {
    const escapedUrl = escapeHtml(block.url)
    mainContentHtml = `
      <div class="arena-modal-embed arena-modal-embed-substack" data-substack-url="${escapedUrl}">
        <span class="arena-loading-spinner" role="status" aria-label="Loading Substack preview"></span>
      </div>
    `
  } else if (block.url && isArenaPdfUrl(block.url)) {
    mainContentHtml = renderPdfModalHtml(block.url)
  } else if (block.url) {
    const externalEmbedMode = block.embedMode ?? 'auto'
    mainContentHtml = renderExternalModalHtml(block, block.url, externalEmbedMode)
  } else if (block.internalSlug) {
    mainContentHtml = `
      <div
        class="arena-modal-internal-host"
        data-block-id="${escapeHtml(block.id)}"
        data-internal-slug="${escapeHtml(block.internalSlug)}"
        data-internal-href="${escapeHtml(block.internalHref || '')}"
        data-internal-hash="${escapeHtml(block.internalHash || '')}"
      >
        <div class="arena-modal-internal-preview grid"></div>
      </div>
    `
  } else {
    mainContentHtml = `<div class="arena-modal-placeholder">No preview available</div>`
  }

  container.innerHTML = `
    <div class="arena-modal-layout">
      <div class="arena-modal-main">
        ${
          displayUrl
            ? `
          <div class="arena-modal-url-bar">
            <button type="button" class="arena-url-copy-button" data-url="${escapedDisplayUrl}" role="button" tabIndex="0" aria-label="Copy URL to clipboard">
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" class="copy-icon">
                <path d="M7.49996 1.80002C4.35194 1.80002 1.79996 4.352 1.79996 7.50002C1.79996 10.648 4.35194 13.2 7.49996 13.2C10.648 13.2 13.2 10.648 13.2 7.50002C13.2 4.352 10.648 1.80002 7.49996 1.80002ZM0.899963 7.50002C0.899963 3.85494 3.85488 0.900024 7.49996 0.900024C11.145 0.900024 14.1 3.85494 14.1 7.50002C14.1 11.1451 11.145 14.1 7.49996 14.1C3.85488 14.1 0.899963 11.1451 0.899963 7.50002Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M13.4999 7.89998H1.49994V7.09998H13.4999V7.89998Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M7.09991 13.5V1.5H7.89991V13.5H7.09991zM10.375 7.49998C10.375 5.32724 9.59364 3.17778 8.06183 1.75656L8.53793 1.24341C10.2396 2.82218 11.075 5.17273 11.075 7.49998 11.075 9.82724 10.2396 12.1778 8.53793 13.7566L8.06183 13.2434C9.59364 11.8222 10.375 9.67273 10.375 7.49998zM3.99969 7.5C3.99969 5.17611 4.80786 2.82678 6.45768 1.24719L6.94177 1.75281C5.4582 3.17323 4.69969 5.32389 4.69969 7.5 4.6997 9.67611 5.45822 11.8268 6.94179 13.2472L6.45769 13.7528C4.80788 12.1732 3.9997 9.8239 3.99969 7.5z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M7.49996 3.95801C9.66928 3.95801 11.8753 4.35915 13.3706 5.19448 13.5394 5.28875 13.5998 5.50197 13.5055 5.67073 13.4113 5.83948 13.198 5.89987 13.0293 5.8056 11.6794 5.05155 9.60799 4.65801 7.49996 4.65801 5.39192 4.65801 3.32052 5.05155 1.97064 5.8056 1.80188 5.89987 1.58866 5.83948 1.49439 5.67073 1.40013 5.50197 1.46051 5.28875 1.62927 5.19448 3.12466 4.35915 5.33063 3.95801 7.49996 3.95801zM7.49996 10.85C9.66928 10.85 11.8753 10.4488 13.3706 9.6135 13.5394 9.51924 13.5998 9.30601 13.5055 9.13726 13.4113 8.9685 13.198 8.90812 13.0293 9.00238 11.6794 9.75643 9.60799 10.15 7.49996 10.15 5.39192 10.15 3.32052 9.75643 1.97064 9.00239 1.80188 8.90812 1.58866 8.9685 1.49439 9.13726 1.40013 9.30601 1.46051 9.51924 1.62927 9.6135 3.12466 10.4488 5.33063 10.85 7.49996 10.85z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
              </svg>
              <svg width="15" height="15" viewBox="-2 -2 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" class="check-icon">
                <use href="#github-check"/>
              </svg>
            </button>
            ${
              block.url
                ? `
              <a href="${escapeHtml(block.url)}" target="_blank" rel="noopener noreferrer" class="arena-modal-link">
                <div class="arena-modal-link-text">${escapedDisplayUrl}</div>
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path fill-rule="evenodd" clip-rule="evenodd" d="M12 13C12.5523 13 13 12.5523 13 12V3C13 2.44771 12.5523 2 12 2H3C2.44771 2 2 2.44771 2 3V6.5C2 6.77614 2.22386 7 2.5 7C2.77614 7 3 6.77614 3 6.5V3H12V12H8.5C8.22386 12 8 12.2239 8 12.5C8 12.7761 8.22386 13 8.5 13H12ZM9 6.5C9 6.5001 9 6.50021 9 6.50031V6.50035V9.5C9 9.77614 8.77614 10 8.5 10C8.22386 10 8 9.77614 8 9.5V7.70711L2.85355 12.8536C2.65829 13.0488 2.34171 13.0488 2.14645 12.8536C1.95118 12.6583 1.95118 12.3417 2.14645 12.1464L7.29289 7H5.5C5.22386 7 5 6.77614 5 6.5C5 6.22386 5.22386 6 5.5 6H8.5C8.56779 6 8.63244 6.01349 8.69139 6.03794C8.74949 6.06198 8.80398 6.09744 8.85143 6.14433C8.94251 6.23434 8.9992 6.35909 8.99999 6.49708L8.99999 6.49738" fill="currentColor"/>
                </svg>
              </a>
            `
                : `
              <span class="arena-modal-link">
                <div class="arena-modal-link-text">${escapedDisplayUrl}</div>
              </span>
            `
            }
          </div>
        `
            : ''
        }
        ${mapHtml}
        <div class="arena-modal-main-content">
          ${mainContentHtml}
        </div>
      </div>
      <div class="arena-modal-sidebar">
        <div class="arena-modal-info">
          <h3 class="arena-modal-title">
            ${block.titleHtml ?? escapeHtml(block.title ?? '')}
          </h3>
          ${
            metadataHtml.length > 0
              ? `
            <div class="arena-modal-meta">
              ${metadataHtml.join('')}
            </div>
          `
              : ''
          }
        </div>
        ${subItemsHtml}
      </div>
    </div>
  `

  return container
}

export async function showModal(blockId: string) {
  const modal = document.getElementById('arena-modal')
  const modalBody = modal?.querySelector('.arena-modal-body') as HTMLElement | null
  if (!modal || !modalBody) return

  const dataEl = document.getElementById(`arena-modal-data-${blockId}`)
  if (!dataEl) {
    console.warn(`Modal data not found for block ${blockId}`)
    return
  }

  const blockEl = document.querySelector(`[data-block-id="${blockId}"]`)
  if (blockEl) {
    currentBlockIndex = parseInt(blockEl.getAttribute('data-block-index') || '0')
  } else {
    // On index page or dynamic modal - disable navigation by setting out of bounds
    currentBlockIndex = 0
  }

  cleanupMaps(modalBody)
  modalBody.innerHTML = ''
  const clonedContent = dataEl.cloneNode(true) as HTMLElement
  clonedContent.style.display = 'block'
  modalBody.appendChild(clonedContent)

  if (window.twttr && typeof window.twttr.ready === 'function') {
    window.twttr.ready((readyTwttr: any) => {
      if (readyTwttr?.widgets?.load) {
        readyTwttr.widgets.load(modalBody)
      }
    })
    // @ts-ignore
  } else if (window.twttr?.widgets?.load) {
    // @ts-ignore
    window.twttr.widgets.load(modalBody)
  }

  hydrateSubstackEmbeds(modalBody)
  hydrateExternalEmbeds(modalBody)
  hydrateInternalHosts(modalBody)
  hydrateMapboxMaps(modalBody)
  mountPdfEmbeds(modalBody)

  const sidebar = modalBody.querySelector('.arena-modal-sidebar') as HTMLElement | null
  const hasConnections = modalBody.querySelector('.arena-modal-connections') !== null
  const collapseBtn = modal?.querySelector('.arena-modal-collapse') as HTMLElement | null

  if (sidebar) {
    let shouldCollapse = !hasConnections

    // Check for sidebar metadata
    if (arenaSearchData) {
      const blockData = arenaSearchData.blocks.find(b => b.id === blockId)
      if (blockData?.metadata?.sidebar) {
        const sidebarValue = blockData.metadata.sidebar.toLowerCase().trim()
        if (sidebarValue === 'false' || sidebarValue === '0') {
          shouldCollapse = true
        }
      }
    }

    if (shouldCollapse) {
      sidebar.classList.add('collapsed')
      collapseBtn?.classList.add('active')
    } else {
      sidebar.classList.remove('collapsed')
      collapseBtn?.classList.remove('active')
    }
  }

  updateNavButtons()
  modal.classList.add('active')
  lockPageScroll()
}

export function closeModal() {
  const modal = document.getElementById('arena-modal')
  if (modal) {
    const modalBody = modal.querySelector('.arena-modal-body') as HTMLElement | null
    if (modalBody) {
      cleanupMaps(modalBody)
      cleanupPdfs(modalBody)
    }
    modal.classList.remove('active')
    unlockPageScroll()
  }
}

export async function navigateBlock(direction: number) {
  const newIndex = currentBlockIndex + direction
  if (newIndex < 0 || newIndex >= totalBlocks) return

  const blocks = Array.from(document.querySelectorAll('.arena-block[data-block-id]'))
  const targetBlock = blocks[newIndex] as HTMLElement
  if (!targetBlock) return

  const blockId = targetBlock.getAttribute('data-block-id')
  if (blockId) {
    await showModal(blockId)
  }
}

function updateNavButtons() {
  const prevBtn = document.querySelector('.arena-modal-prev') as HTMLButtonElement
  const nextBtn = document.querySelector('.arena-modal-next') as HTMLButtonElement

  if (prevBtn) {
    prevBtn.disabled = currentBlockIndex === 0
    prevBtn.style.opacity = currentBlockIndex === 0 ? '0.3' : '1'
  }

  if (nextBtn) {
    nextBtn.disabled = currentBlockIndex >= totalBlocks - 1
    nextBtn.style.opacity = currentBlockIndex >= totalBlocks - 1 ? '0.3' : '1'
  }
}

export function handleCopyButton(button: HTMLElement) {
  const targetUrl = button.getAttribute('data-url')
  if (!targetUrl) {
    return
  }

  navigator.clipboard.writeText(targetUrl).then(
    () => {
      button.classList.add('check')
      setTimeout(() => {
        button.classList.remove('check')
      }, 2000)
    },
    error => console.error(error),
  )
}

// Search functionality - JSON-based
interface ArenaBlockSearchable {
  id: string
  channelSlug: string
  channelName: string
  content: string
  title?: string
  titleHtml?: string
  blockHtml?: string
  url?: string
  highlighted: boolean
  embedHtml?: string
  metadata?: Record<string, string>
  coordinates?: { lon: number; lat: number }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  tags?: string[]
  subItems?: ArenaBlockSearchable[]
  hasModalInDom: boolean
  embedMode?: ArenaExternalEmbedMode
}

interface ArenaChannelSearchable {
  id: string
  name: string
  slug: string
  blockCount: number
}

interface ArenaSearchIndex {
  version: string
  blocks: ArenaBlockSearchable[]
  channels: ArenaChannelSearchable[]
}

interface SearchIndexItem {
  blockId: string
  channelSlug?: string
  channelName?: string
  title: string
  content: string
  highlighted: boolean
  hasModalInDom: boolean
  tags?: string[]
  matchedTags?: string[]
}

let searchIndex: SearchIndexItem[] = []
let searchDebounceTimer: number | undefined
let arenaSearchData: ArenaSearchIndex | null = null

export const getSearchIndex = () => searchIndex
export const setSearchIndex = (index: SearchIndexItem[]) => {
  searchIndex = index
}

// Fetch arena search index JSON
async function fetchArenaSearchIndex(): Promise<ArenaSearchIndex | null> {
  if (arenaSearchData) return arenaSearchData

  try {
    const response = await fetch('/static/arena-search.json')
    if (!response.ok) {
      console.warn(`Failed to fetch arena search index: ${response.status}`)
      return null
    }
    arenaSearchData = (await response.json()) as ArenaSearchIndex
    return arenaSearchData
  } catch (error) {
    console.error('Error fetching arena search index:', error)
    return null
  }
}

export async function buildSearchIndex(scope: SearchScope): Promise<SearchIndexItem[]> {
  const index: SearchIndexItem[] = []

  if (scope === 'channel') {
    // For channel pages, fetch JSON and filter by current channel
    const data = await fetchArenaSearchIndex()
    if (!data) return index

    const currentSlug = document.body?.dataset.slug || ''
    const channelSlug = currentSlug.replace(/^arena\//, '')

    data.blocks
      .filter(block => block.channelSlug === channelSlug)
      .forEach(block => {
        index.push({
          blockId: block.id,
          channelSlug: block.channelSlug,
          channelName: block.channelName,
          title: block.title || block.content,
          content: block.content,
          highlighted: block.highlighted,
          hasModalInDom: block.hasModalInDom,
          tags: block.tags,
        })
      })
  } else {
    // For index page, use all blocks from JSON
    const data = await fetchArenaSearchIndex()
    if (!data) return index

    data.blocks.forEach(block => {
      index.push({
        blockId: block.id,
        channelSlug: block.channelSlug,
        channelName: block.channelName,
        title: block.title || block.content,
        content: block.content,
        highlighted: block.highlighted,
        hasModalInDom: block.hasModalInDom,
        tags: block.tags,
      })
    })
  }

  return index
}

function parseSearchQuery(query: string): { tagQueries: string[]; regularTerms: string[] } {
  const tagQueries: string[] = []
  let remainingQuery = query

  const tagPatternRegex =
    /#([^#]+?)(?=\s*#|\s*r\/|$)|r\/([^\s#]+(?:\s+[^\s#]+)*?)(?=\s*#|\s*r\/|$)/g
  let match

  while ((match = tagPatternRegex.exec(query)) !== null) {
    const tagText = (match[1] || match[2] || '').trim()
    if (tagText.length > 0) {
      tagQueries.push(tagText)
      remainingQuery = remainingQuery.replace(match[0], ' ')
    }
  }

  const regularTerms = remainingQuery
    .trim()
    .split(/\s+/)
    .filter(term => term.length > 0)

  return { tagQueries, regularTerms }
}

export function performSearch(query: string, index: SearchIndexItem[]): SearchIndexItem[] {
  const lowerQuery = query.toLowerCase().trim()
  if (lowerQuery.length < 2) return []

  const { tagQueries, regularTerms } = parseSearchQuery(lowerQuery)
  const tokens = tokenizeTerm(regularTerms.join(' '))

  const scoredResults = index
    .map(item => {
      const lowerTitle = item.title.toLowerCase()
      const lowerContent = item.content.toLowerCase()
      const itemTags = item.tags || []
      const matchedTags: string[] = []

      let score = 0
      let titleMatchCount = 0
      let contentMatchCount = 0
      let tagMatchCount = 0

      // Tag-specific queries (# or r/ prefix)
      for (const tagQuery of tagQueries) {
        const lowerTagQuery = tagQuery.toLowerCase()
        for (const tag of itemTags) {
          const lowerTag = tag.toLowerCase()
          if (lowerTag === lowerTagQuery) {
            score += 25
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          } else if (lowerTag.includes(lowerTagQuery)) {
            score += 20
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          }
        }
      }

      // Regular terms match tags
      for (const term of regularTerms) {
        const lowerTerm = term.toLowerCase()
        for (const tag of itemTags) {
          const lowerTag = tag.toLowerCase()
          if (lowerTag === lowerTerm) {
            score += 18
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          } else if (lowerTag.includes(lowerTerm)) {
            score += 15
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          }
        }
      }

      // Check each token for title and content matches
      for (const token of tokens) {
        const tokenLower = token.toLowerCase()

        if (lowerTitle.includes(tokenLower)) {
          titleMatchCount++
          score += 10
        }

        if (lowerContent.includes(tokenLower)) {
          contentMatchCount++
          score += 5
        }
      }

      // Boost for exact phrase match in title/content
      if (regularTerms.length > 0) {
        const regularQuery = regularTerms.join(' ').toLowerCase()
        if (lowerTitle.includes(regularQuery)) {
          score += 20
        }
        if (lowerContent.includes(regularQuery)) {
          score += 10
        }
      }

      // Boost for highlighted items
      if (item.highlighted) {
        score += 3
      }

      // Must have at least one match
      if (score === 0) return null

      const resultItem = { ...item }
      if (matchedTags.length > 0) {
        resultItem.matchedTags = matchedTags
      }

      return { item: resultItem, score, titleMatchCount, contentMatchCount, tagMatchCount }
    })
    .filter((result): result is NonNullable<typeof result> => result !== null)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score

      if (b.tagMatchCount !== a.tagMatchCount) {
        return b.tagMatchCount - a.tagMatchCount
      }

      if (b.titleMatchCount !== a.titleMatchCount) {
        return b.titleMatchCount - a.titleMatchCount
      }

      return b.contentMatchCount - a.contentMatchCount
    })

  return scoredResults.map(result => result.item)
}

export function renderSearchResults(
  results: SearchIndexItem[],
  scope: SearchScope,
  searchQuery: string = '',
) {
  const container = document.getElementById('arena-search-container')
  if (!container) return

  if (results.length === 0) {
    container.innerHTML = '<div class="arena-search-no-results">no results found</div>'
    container.classList.add('active')
    return
  }

  const fragment = document.createDocumentFragment()
  results.forEach((result, idx) => {
    const resultItem = document.createElement('div')
    resultItem.className = 'arena-search-result-item'
    resultItem.setAttribute('data-block-id', result.blockId)
    if (result.channelSlug) {
      resultItem.setAttribute('data-channel-slug', result.channelSlug)
    }
    resultItem.dataset.index = String(idx)
    resultItem.tabIndex = -1
    resultItem.setAttribute('role', 'option')
    if (!resultItem.id) {
      const uniqueId = `arena-search-${result.blockId}`
      resultItem.id = uniqueId
    }

    const title = document.createElement('div')
    title.className = 'arena-search-result-title'
    appendHighlightedText(title, searchQuery, result.title)
    resultItem.appendChild(title)

    if (result.matchedTags && result.matchedTags.length > 0) {
      const tagsContainer = document.createElement('ul')
      tagsContainer.className = 'arena-search-result-tags'

      result.matchedTags.forEach(tag => {
        const tagBadge = document.createElement('li')
        tagBadge.className = 'arena-search-result-tag-badge'
        appendHighlightedText(tagBadge, searchQuery, tag)
        tagsContainer.appendChild(tagBadge)
      })

      resultItem.appendChild(tagsContainer)
    }

    if (scope === 'index' && result.channelName) {
      const badge = document.createElement('span')
      badge.className = 'arena-search-result-channel-badge'
      badge.textContent = result.channelName
      resultItem.appendChild(badge)
    }

    fragment.appendChild(resultItem)
  })

  container.innerHTML = ''
  container.appendChild(fragment)
  container.classList.add('active')
}

const getSearchResults = () =>
  Array.from(
    document.querySelectorAll<HTMLElement>('#arena-search-container .arena-search-result-item'),
  )

export const resetActiveResultHighlight = () => {
  activeResultIndex = null
  if (searchInput) {
    searchInput.removeAttribute('aria-activedescendant')
  }
  getSearchResults().forEach(result => result.classList.remove('active'))
}

export const setActiveResult = (index: number | null, options?: SearchResultOptions) => {
  const results = getSearchResults()
  if (results.length === 0) {
    resetActiveResultHighlight()
    if (options?.focus && searchInput) {
      searchInput.focus()
    }
    return
  }

  if (index === null) {
    resetActiveResultHighlight()
    if (options?.focus && searchInput) {
      searchInput.focus()
    }
    return
  }

  const clamped = Math.max(0, Math.min(index, results.length - 1))
  results.forEach((result, idx) => {
    result.classList.toggle('active', idx === clamped)
  })

  const target = results[clamped]
  activeResultIndex = clamped
  if (!target.id) {
    const fallbackId = target.getAttribute('data-block-id') || `arena-result-${clamped}`
    target.id = `arena-search-${fallbackId}`
  }
  if (searchInput) {
    searchInput.setAttribute('aria-activedescendant', target.id)
  }

  if (options?.focus !== false) {
    target.focus()
  }

  if (options?.scroll !== false) {
    target.scrollIntoView({ block: 'nearest' })
  }
}

export const wireSearchResultsInteractions = () => {
  const items = getSearchResults()
  items.forEach(item => {
    const idx = Number.parseInt(item.dataset.index ?? '', 10)
    if (!Number.isInteger(idx)) return

    const onFocus = () => setActiveResult(idx, { focus: false, scroll: false })
    const onMouseEnter = () => setActiveResult(idx, { focus: false, scroll: false })

    item.addEventListener('focus', onFocus)
    item.addEventListener('mouseenter', onMouseEnter)

    addCleanup(() => {
      item.removeEventListener('focus', onFocus)
      item.removeEventListener('mouseenter', onMouseEnter)
    })
  })
}

export const focusSearchInput = (prefill?: string) => {
  if (!searchInput) return
  if (typeof prefill === 'string') {
    searchInput.value = prefill
    searchInput.dispatchEvent(new Event('input', { bubbles: true }))
  }
  resetActiveResultHighlight()
  const valueLength = searchInput.value.length
  searchInput.focus()
  try {
    if (valueLength > 0) {
      searchInput.select()
    } else {
      searchInput.setSelectionRange(valueLength, valueLength)
    }
  } catch {}
}

export const clearSearchState = (options?: { blur?: boolean }) => {
  if (searchInput) {
    searchInput.value = ''
    if (options?.blur !== false) {
      searchInput.blur()
    }
    searchInput.removeAttribute('aria-activedescendant')
  }
  resetActiveResultHighlight()
  closeSearchDropdown()
}

export function closeSearchDropdown() {
  const container = document.getElementById('arena-search-container')
  if (container) {
    container.classList.remove('active')
    container.innerHTML = ''
  }
}

export const mountArena = (dispatch: (event: ArenaEvent) => void) => {
  const cleanups: Array<() => void> = []
  const previousRegister = registerCleanup
  const register = (cleanup: () => void) => {
    cleanups.push(cleanup)
  }
  registerCleanup = register

  totalBlocks = document.querySelectorAll('[data-block-id][data-block-index]').length

  const channelPage = document.querySelector<HTMLElement>('.arena-channel-page')
  const blockCollection = document.getElementById('arena-block-collection') as HTMLElement | null
  const viewToggleButtons = channelPage
    ? Array.from(channelPage.querySelectorAll<HTMLButtonElement>('.arena-view-toggle-button'))
    : []

  type ArenaViewMode = 'grid' | 'list'

  const applyViewMode = (mode: ArenaViewMode) => {
    if (!channelPage || !blockCollection) return
    const normalized: ArenaViewMode = mode === 'list' ? 'list' : 'grid'
    channelPage.dataset.viewMode = normalized
    blockCollection.dataset.viewMode = normalized
    viewToggleButtons.forEach(button => {
      const targetMode = (button.dataset.viewMode as ArenaViewMode) || 'grid'
      const isActive = targetMode === normalized
      button.classList.toggle('active', isActive)
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false')
    })
    totalBlocks = channelPage.querySelectorAll('[data-block-id][data-block-index]').length
  }

  if (channelPage && blockCollection && viewToggleButtons.length > 0) {
    const initialMode = (blockCollection.dataset.viewMode as ArenaViewMode | undefined) ?? 'grid'
    applyViewMode(initialMode)

    viewToggleButtons.forEach(button => {
      const desiredMode = (button.dataset.viewMode as ArenaViewMode) || 'grid'
      const onToggleClick = () => applyViewMode(desiredMode)
      button.addEventListener('click', onToggleClick)
      addCleanup(() => button.removeEventListener('click', onToggleClick))
    })
  }

  // Build search index
  searchInput = document.querySelector<HTMLInputElement>('.arena-search-input')
  activeResultIndex = null

  if (searchInput) {
    const input = searchInput
    const scope = input.getAttribute('data-search-scope') as SearchScope
    dispatch({ type: 'search.index.request', scope })

    if (searchDebounceTimer) {
      window.clearTimeout(searchDebounceTimer)
    }

    const onSearchInput = (e: Event) => {
      const target = e.target as HTMLInputElement
      const query = target.value

      window.clearTimeout(searchDebounceTimer)
      searchDebounceTimer = window.setTimeout(() => {
        dispatch({ type: 'ui.search.query', query, scope })
      }, 300)
    }

    input.addEventListener('input', onSearchInput)
    addCleanup(() => input.removeEventListener('input', onSearchInput))
    addCleanup(() => {
      if (searchDebounceTimer) {
        window.clearTimeout(searchDebounceTimer)
        searchDebounceTimer = undefined
      }
    })
  }

  const onClick = async (e: MouseEvent) => {
    const target = e.target as HTMLElement
    const isArenaChannelPage = () => {
      const slug = document.body?.dataset.slug || ''
      return slug.startsWith('arena/') && slug !== 'arena'
    }

    const internalLink = target.closest('.arena-modal-body a.internal') as HTMLAnchorElement | null
    if (internalLink) {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
        return
      }

      e.preventDefault()

      // Check if this is a wikilink trail anchor - open in stacked notes
      const isWikilinkTrail = internalLink.classList.contains('arena-wikilink-trail-anchor')

      if (isWikilinkTrail && typeof window.stacked !== 'undefined') {
        try {
          const destination = new URL(internalLink.href)
          await window.stacked.navigate(destination)
          return
        } catch (err) {
          console.error('Failed to open in stacked notes:', err)
        }
      }

      // Regular internal link - close modal and navigate
      dispatch({ type: 'ui.modal.close' })
      try {
        const destination = new URL(internalLink.href)
        if (typeof window.spaNavigate === 'function') {
          window.spaNavigate(destination)
        } else {
          window.location.assign(destination)
        }
      } catch (err) {
        console.error(err)
        window.location.assign(internalLink.href)
      }
      return
    }

    const copyButton = target.closest('button.arena-url-copy-button') as HTMLElement | null
    if (copyButton) {
      e.preventDefault()
      e.stopPropagation()
      dispatch({ type: 'ui.copy', button: copyButton })
      return
    }

    const blockClickable = target.closest('.arena-block-clickable')
    if (blockClickable) {
      const blockEl = blockClickable.closest('.arena-block')
      const blockId = blockEl?.getAttribute('data-block-id')
      if (blockId) {
        e.preventDefault()

        // Check if this is an arxiv block without notes and redirect instead of opening modal
        if (arenaSearchData) {
          const blockData = arenaSearchData.blocks.find(b => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              window.open(blockData.url, '_blank', 'noopener,noreferrer')
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        dispatch({ type: 'ui.modal.open', blockId })
      }
      return
    }

    const previewItem = target.closest('.arena-channel-row-preview-item[data-block-id]')
    if (previewItem) {
      const blockId = (previewItem as HTMLElement).getAttribute('data-block-id')

      if (isArenaChannelPage()) {
        // On channel pages: check if arxiv block without notes and redirect instead of modal
        if (blockId && arenaSearchData) {
          const blockData = arenaSearchData.blocks.find(b => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              e.preventDefault()
              window.open(blockData.url, '_blank', 'noopener,noreferrer')
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        if (blockId) {
          e.preventDefault()
          dispatch({ type: 'ui.modal.open', blockId })
        }
      } else {
        // On Arena index page, clicking a preview should navigate to the channel
        const channelRow = previewItem.closest('.arena-channel-row') as HTMLElement | null
        const headerLink = channelRow?.querySelector(
          '.arena-channel-row-header a[href]',
        ) as HTMLAnchorElement | null
        if (headerLink) {
          e.preventDefault()
          headerLink.click()
        }
      }
      return
    }

    // Click anywhere on a channel row should navigate to the header link
    const channelRow = target.closest('.arena-channel-row') as HTMLElement | null
    if (channelRow) {
      // If the user clicked a real interactive element, let it handle itself
      if (target.closest('a,button,[role=button],input,textarea,select,summary')) {
        return
      }
      const headerLink = channelRow.querySelector(
        '.arena-channel-row-header a[href]',
      ) as HTMLAnchorElement | null
      if (headerLink) {
        e.preventDefault()
        // prefer native navigation so SPA router (if any) can hook the event
        headerLink.click()
      }
      return
    }

    if (target.closest('.arena-modal-prev')) {
      dispatch({ type: 'ui.block.navigate', direction: -1 })
      return
    }

    if (target.closest('.arena-modal-next')) {
      dispatch({ type: 'ui.block.navigate', direction: 1 })
      return
    }

    if (target.closest('.arena-modal-collapse')) {
      const modal = document.getElementById('arena-modal')
      const sidebar = modal?.querySelector('.arena-modal-sidebar') as HTMLElement | null
      const collapseBtn = target.closest('.arena-modal-collapse') as HTMLElement | null
      if (sidebar) {
        sidebar.classList.toggle('collapsed')
        collapseBtn?.classList.toggle('active')

        // Resize maps after layout change - wait for CSS transition
        const modalBody = modal?.querySelector('.arena-modal-body') as HTMLElement | null
        if (modalBody) {
          const resizeMaps = () => {
            modalBody
              .querySelectorAll<HTMLElement>(".arena-modal-map[data-map-initialized='1']")
              .forEach(mapNode => {
                const map = mapInstances.get(mapNode)
                if (map) {
                  try {
                    map.resize()
                  } catch (error) {
                    console.error(error)
                  }
                }
              })
          }

          const onTransitionEnd = (event: TransitionEvent) => {
            if (event.target === sidebar && event.propertyName === 'width') {
              sidebar.removeEventListener('transitionend', onTransitionEnd)
              requestAnimationFrame(resizeMaps)
            }
          }

          sidebar.addEventListener('transitionend', onTransitionEnd)
          setTimeout(() => {
            sidebar.removeEventListener('transitionend', onTransitionEnd)
            requestAnimationFrame(resizeMaps)
          }, 500)
        }
      }
      return
    }

    // Handle search result clicks with smart navigation based on hasModalInDom
    const searchResultItem = target.closest('.arena-search-result-item') as HTMLElement | null
    if (searchResultItem) {
      const blockId = searchResultItem.getAttribute('data-block-id')
      const channelSlug = searchResultItem.getAttribute('data-channel-slug')

      if (blockId) {
        e.preventDefault()
        dispatch({ type: 'ui.search.clear', blur: true })

        // Check if this is an arxiv block without notes and redirect instead of opening modal
        if (arenaSearchData) {
          const blockData = arenaSearchData.blocks.find(b => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              window.open(blockData.url, '_blank', 'noopener,noreferrer')
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        // Find the result in search index to check hasModalInDom flag
        const resultData = searchIndex.find(item => item.blockId === blockId)

        if (resultData?.hasModalInDom) {
          // Block has prerendered modal in DOM - show it instantly
          dispatch({ type: 'ui.modal.open', blockId })
        } else {
          // Block doesn't have modal in DOM - create it dynamically from JSON
          const existingModal = document.getElementById(`arena-modal-data-${blockId}`)

          if (!existingModal && channelSlug) {
            // Ensure search data is loaded
            if (!arenaSearchData) {
              await fetchArenaSearchIndex()
            }

            if (!arenaSearchData) {
              console.error('Failed to load arena search data')
              return
            }

            // Find full block data in search JSON
            const fullBlockData = arenaSearchData.blocks.find(b => b.id === blockId)

            if (fullBlockData) {
              // Create modal data element from JSON
              const modalDataEl = createModalDataFromJson(fullBlockData, channelSlug)

              // Append to body so getElementById can find it easily
              document.body.appendChild(modalDataEl)

              // Now show the modal
              dispatch({ type: 'ui.modal.open', blockId })
            } else if (channelSlug) {
              // Fallback: navigate to channel page if we can't find the data
              const currentSlug = document.body?.dataset.slug || ''
              const targetChannelSlug = `arena/${channelSlug}`

              if (currentSlug === targetChannelSlug) {
                dispatch({ type: 'ui.modal.open', blockId })
              } else {
                const targetUrl = new URL(
                  `/${['arena', ...channelSlug.split('/')].map(encodeURIComponent).join('/')}`,
                  window.location.origin,
                )
                targetUrl.hash = blockId
                window.location.assign(targetUrl)
              }
            } else {
              // Last resort: try to show modal anyway
              dispatch({ type: 'ui.modal.open', blockId })
            }
          } else {
            // Modal data already exists, just show it
            dispatch({ type: 'ui.modal.open', blockId })
          }
        }
      }
      return
    }

    if (target.closest('.arena-modal-close') || target.classList.contains('arena-block-modal')) {
      dispatch({ type: 'ui.modal.close' })
    }
  }

  const onKey = (e: KeyboardEvent) => {
    const key = e.key.toLowerCase()
    if ((e.metaKey || e.ctrlKey) && key === 'k') {
      e.preventDefault()
      if (e.shiftKey) {
        dispatch({ type: 'ui.search.focus', prefill: '#' })
      } else {
        dispatch({ type: 'ui.search.focus' })
      }
      return
    }

    if ((e.metaKey || e.ctrlKey) && key === 'b') {
      const modal = document.getElementById('arena-modal')
      if (modal?.classList.contains('active')) {
        e.preventDefault()
        const sidebar = modal.querySelector('.arena-modal-sidebar') as HTMLElement | null
        const collapseBtn = modal.querySelector('.arena-modal-collapse') as HTMLElement | null
        if (sidebar) {
          sidebar.classList.toggle('collapsed')
          collapseBtn?.classList.toggle('active')

          // Resize maps after layout change - wait for CSS transition
          const modalBody = modal.querySelector('.arena-modal-body') as HTMLElement | null
          if (modalBody) {
            const resizeMaps = () => {
              modalBody
                .querySelectorAll<HTMLElement>(".arena-modal-map[data-map-initialized='1']")
                .forEach(mapNode => {
                  const map = mapInstances.get(mapNode)
                  if (map) {
                    try {
                      map.resize()
                    } catch (error) {
                      console.error(error)
                    }
                  }
                })
            }

            const onTransitionEnd = (event: TransitionEvent) => {
              if (event.target === sidebar && event.propertyName === 'width') {
                sidebar.removeEventListener('transitionend', onTransitionEnd)
                requestAnimationFrame(resizeMaps)
              }
            }

            sidebar.addEventListener('transitionend', onTransitionEnd)
            setTimeout(() => {
              sidebar.removeEventListener('transitionend', onTransitionEnd)
              requestAnimationFrame(resizeMaps)
            }, 100)
          }
        }
        return
      }
    }

    const results = getSearchResults()
    const searchContainer = document.getElementById('arena-search-container')
    const searchOpen = Boolean(searchContainer && searchContainer.classList.contains('active'))
    const resultFocused = document.activeElement?.classList.contains('arena-search-result-item')
    const inputFocused = document.activeElement === searchInput

    if (searchOpen && results.length > 0) {
      if (
        key === 'ArrowDown' ||
        (!e.shiftKey && key === 'Tab') ||
        (key === 'n' && (e.ctrlKey || e.metaKey))
      ) {
        e.preventDefault()
        if (
          !e.shiftKey &&
          key === 'Tab' &&
          resultFocused &&
          activeResultIndex === results.length - 1
        ) {
          dispatch({
            type: 'ui.search.result.activate',
            index: null,
            options: { focus: true, scroll: false },
          })
        } else if (inputFocused || activeResultIndex === null) {
          dispatch({ type: 'ui.search.result.activate', index: 0 })
        } else {
          const nextIndex = Math.min((activeResultIndex ?? -1) + 1, results.length - 1)
          dispatch({ type: 'ui.search.result.activate', index: nextIndex })
        }
        return
      }

      if (
        key === 'ArrowUp' ||
        (e.shiftKey && key === 'Tab') ||
        (key === 'p' && (e.ctrlKey || e.metaKey))
      ) {
        e.preventDefault()
        if (!resultFocused || activeResultIndex === null) {
          dispatch({ type: 'ui.search.result.activate', index: results.length - 1 })
        } else if (activeResultIndex <= 0) {
          dispatch({
            type: 'ui.search.result.activate',
            index: null,
            options: { focus: true, scroll: false },
          })
        } else {
          dispatch({ type: 'ui.search.result.activate', index: activeResultIndex - 1 })
        }
        return
      }

      if (key === 'enter') {
        e.preventDefault()
        if (resultFocused) {
          document.activeElement?.dispatchEvent(new MouseEvent('click', { bubbles: true }))
        } else if (inputFocused && results.length > 0) {
          // Open first result if Enter pressed in input
          results[0]?.dispatchEvent(new MouseEvent('click', { bubbles: true }))
        }
        return
      }
    }

    if (key === 'escape') {
      if (searchContainer && searchContainer.classList.contains('active')) {
        dispatch({ type: 'ui.search.close' })
      } else {
        dispatch({ type: 'ui.modal.close' })
      }
      return
    }

    // Handle PDF page navigation with Shift+Arrow
    if (e.shiftKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      const modal = document.getElementById('arena-modal')
      const modalBody = modal?.querySelector('.arena-modal-body')
      const pdfViewer = modalBody?.querySelector(".pdf-embed[data-pdf-status='loaded']")

      if (pdfViewer) {
        e.preventDefault()
        const controls = pdfViewer.querySelector('.pdf-embed-toolbar')
        if (controls) {
          if (e.key === 'ArrowLeft') {
            const prevBtn = controls.querySelector('.pdf-embed-prev') as HTMLButtonElement
            if (prevBtn && !prevBtn.disabled) {
              prevBtn.click()
            }
          } else {
            const nextBtn = controls.querySelector('.pdf-embed-next') as HTMLButtonElement
            if (nextBtn && !nextBtn.disabled) {
              nextBtn.click()
            }
          }
        }
        return
      }
    }

    if (e.key === 'ArrowLeft') {
      dispatch({ type: 'ui.block.navigate', direction: -1 })
    } else if (e.key === 'ArrowRight') {
      dispatch({ type: 'ui.block.navigate', direction: 1 })
    }
  }

  document.addEventListener('click', onClick)
  document.addEventListener('keydown', onKey)
  addCleanup(() => document.removeEventListener('click', onClick))
  addCleanup(() => document.removeEventListener('keydown', onKey))

  return () => {
    const pending = cleanups.slice()
    cleanups.length = 0
    for (const cleanup of pending) {
      cleanup()
    }
    if (registerCleanup === register) {
      registerCleanup = previousRegister
    }
    searchInput = null
    activeResultIndex = null
  }
}
