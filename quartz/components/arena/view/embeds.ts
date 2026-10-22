import { fetchCanonical } from '../../../util/fetch-canonical'
import { normalizeRelativeURLs } from '../../../util/path'
import { applyMonochromeMapPalette, loadMapbox } from '../../scripts/mapbox-client'

const p = new DOMParser()

export const mapInstances = new WeakMap<HTMLElement, any>()

function renderMapFallback(node: HTMLElement, message: string) {
  node.dataset.mapStatus = 'error'
  node.classList.add('arena-map-error')
  node.textContent = message
}

export function cleanupMaps(root: HTMLElement) {
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

export function hydrateMapboxMaps(root: HTMLElement) {
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
          markerEl.style.cursor = 'var(--cursor-pointer, pointer)'

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

export function hydrateSubstackEmbeds(root: HTMLElement) {
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
      preview.tabIndex = 0
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

export function hydrateInternalHosts(root: HTMLElement) {
  const hosts = root.querySelectorAll<HTMLElement>('.arena-modal-internal-host[data-internal-href]')
  hosts.forEach(host => {
    void hydrateInternalHost(host)
  })
}

export function mountPdfEmbeds(root: HTMLElement) {
  for (const embed of root.querySelectorAll<HTMLElement>('[data-arena-pdf-src]')) {
    const src = embed.dataset.arenaPdfSrc
    if (!src) continue
    embed.dataset.pdfSrc = src
    delete embed.dataset.arenaPdfSrc
  }
  window.quartzPdfEmbeds?.mount(root)
}

export function cleanupPdfs(root: HTMLElement) {
  window.quartzPdfEmbeds?.cleanup(root)
}
