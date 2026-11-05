import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { locus, passageBlocks } from '../../../util/latent/markup'
import { titleText } from '../../../util/latent/title'
import { rowHref, type LatentChunk, type LatentDoc } from '../../../util/latent/types'
import { partOf, previewOf, textOf, fetchJson, type Runtime } from './runtime'

const SVG = 'http://www.w3.org/2000/svg'

function mapBox(svg: SVGSVGElement, box: number[], hit: boolean): void {
  const rect = document.createElementNS(SVG, 'rect')
  const [x, y, width, height] = box
  for (const [key, value] of Object.entries({ x, y, width, height }))
    rect.setAttribute(key, String(value))
  if (hit) rect.dataset.hit = ''
  svg.append(rect)
}

async function chunkMap(
  rt: Runtime,
  doc: LatentDoc,
  chunk: LatentChunk,
  first: LatentChunk[],
): Promise<SVGSVGElement> {
  const svg = document.createElementNS(SVG, 'svg')
  svg.setAttribute('viewBox', '0 0 1000 1300')
  svg.setAttribute('aria-hidden', 'true')
  if (doc.s === 'pdf' && doc.sha && chunk.p) {
    let regions = rt.regions.get(doc.sha)
    if (!regions) {
      regions = fetchJson<{ pages: Record<string, { bbox: number[]; text: string }[]> }>(
        rt,
        `/embeddings/pdf-regions/${doc.sha}.json`,
      )
        .then(data => data.pages)
        .catch(() => null)
      rt.regions.set(doc.sha, regions)
    }
    const boxes = (await regions)?.[String(chunk.p)] ?? []
    const words = new Set(chunk.text.toLowerCase().match(/[a-z0-9]{3,}/g) ?? [])
    let best = -1
    let score = 0
    boxes.forEach((box, i) => {
      const tokens = new Set(box.text.toLowerCase().match(/[a-z0-9]{3,}/g) ?? [])
      const overlap = [...tokens].filter(word => words.has(word)).length / Math.max(1, tokens.size)
      if (overlap > score) {
        best = i
        score = overlap
      }
    })
    boxes.forEach(({ bbox: [x0, y0, x1, y1] }, i) =>
      mapBox(svg, [x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0)], i === best && score >= 0.3),
    )
    if (boxes.length) return svg
  }
  // The document range gives every chunk a position without loading its other text parts.
  const total = Math.max(1, doc.n)
  const bins = Math.min(140, total)
  const columns = Math.ceil(bins / 70)
  const rows = Math.ceil(bins / columns)
  const pitch = 1160 / rows
  const width = 840 / columns
  for (let i = 0; i < bins; i++) {
    const lo = Math.floor((i * total) / bins)
    const hi = Math.floor(((i + 1) * total) / bins)
    const known = total === bins ? first[i] : undefined
    const fraction = known ? Math.min(1, Math.max(0.25, known.text.length / 180)) : 0.8
    const offset = chunk.r - doc.r0
    mapBox(
      svg,
      [
        80 + Math.floor(i / rows) * width,
        70 + (i % rows) * pitch,
        Math.max(3, width * fraction - 8),
        Math.max(3, pitch * 0.65),
      ],
      offset >= lo && offset < hi,
    )
  }
  return svg
}

/** One top-layer preview, fetched on pointer dwell or keyboard focus and cancelled by navigation. */
export function bindChunkPreview(rt: Runtime): void {
  const popup = document.createElement('div')
  popup.className = 'lat-chunk-preview'
  popup.setAttribute('popover', 'manual')
  popup.setAttribute('role', 'region')
  popup.setAttribute('aria-label', 'chunk preview')
  popup.id = 'lat-chunk-preview'
  rt.links.append(popup)
  let anchor: HTMLElement | null = null
  let pending: HTMLElement | null = null
  let wanted = -1
  let timer = 0
  let closeTimer = 0
  let sequence = 0
  const close = () => {
    window.clearTimeout(timer)
    window.clearTimeout(closeTimer)
    sequence++
    anchor?.removeAttribute('aria-details')
    anchor?.removeAttribute('aria-expanded')
    anchor = null
    pending = null
    wanted = -1
    if (popup.matches(':popover-open')) popup.hidePopover()
  }
  const position = () => {
    if (!anchor?.isConnected) {
      close()
      return
    }
    const box = anchor.getBoundingClientRect()
    const viewportWidth = document.documentElement.clientWidth
    const width = Math.min(560, viewportWidth - 16)
    popup.style.width = `${width}px`
    popup.style.left = `${Math.max(8, Math.min(viewportWidth - width - 8, box.left - width / 2 + box.width / 2))}px`
    popup.style.top = `${Math.max(8, Math.min(window.innerHeight - popup.offsetHeight - 8, box.bottom + 6))}px`
  }
  const show = async (target: HTMLElement) => {
    pending = null
    window.clearTimeout(closeTimer)
    const row = Number(target.dataset.latChunk)
    if (row === wanted && popup.matches(':popover-open')) {
      if (anchor !== target) {
        anchor?.removeAttribute('aria-details')
        anchor?.removeAttribute('aria-expanded')
        anchor = target
        popup.hidePopover()
        popup.showPopover({ source: target })
        target.setAttribute('aria-details', popup.id)
        target.setAttribute('aria-expanded', 'true')
      }
      position()
      return
    }
    close()
    anchor = target
    wanted = row
    const current = ++sequence
    const doc = rt.docByRow[row]
    if (!doc) return
    popup.replaceChildren()
    popup.dataset.row = String(row)
    const head = document.createElement('div')
    head.className = 'lat-chunk-preview-head'
    head.textContent = `${titleText(doc.t)} · chunk ${row - doc.r0 + 1}`
    const text = document.createElement('div')
    text.className = 'lat-chunk-preview-text'
    text.textContent = 'loading chunk…'
    popup.append(head, text)
    popup.showPopover({ source: target })
    target.setAttribute('aria-details', popup.id)
    target.setAttribute('aria-expanded', 'true')
    position()
    const stale = () => rt.signal.aborted || sequence !== current || !target.isConnected
    try {
      const [preview, document] = await Promise.all([previewOf(rt, row), textOf(rt, doc.d)])
      if (stale()) return
      const part = Math.floor((row - doc.r0) / Math.max(1, document.part))
      const chunks = await partOf(rt, doc.d, part)
      if (stale()) return
      const chunk = chunks.find(chunk => chunk.r === row)
      if (!chunk) throw new Error('chunk text unavailable')
      head.textContent = `${titleText(doc.t)} · ${locus(preview) || `chunk ${row - doc.r0 + 1}`}`
      text.innerHTML = toHtml(
        h('div', passageBlocks(chunk.text, { markdown: chunk.m === 1 })) as never,
      )
      const map = window.document.createElement('a')
      map.className = 'lat-chunk-map'
      map.href = `?focus=${row}`
      map.dataset.latFocus = String(row)
      map.dataset.kind = doc.s
      map.setAttribute('aria-label', `read chunk ${row - doc.r0 + 1}`)
      const svg = await chunkMap(rt, doc, chunk, document.chunks)
      if (stale()) return
      map.append(svg)
      popup.append(map)
      if (doc.s === 'pdf') {
        const read = window.document.createElement('a')
        read.className = 'lat-chunk-read'
        read.href = rowHref(doc, preview.p)
        read.dataset.noPopover = 'true'
        read.textContent = '/read →'
        popup.append(read)
      }
      const features = window.document.createElement('div')
      features.className = 'lat-link-why'
      features.dataset.row = String(row)
      popup.append(features)
      rt.root.dispatchEvent(
        new CustomEvent('latent:chunk-preview', { detail: { row, slot: features } }),
      )
      position()
    } catch (error) {
      if (!stale()) {
        text.textContent = error instanceof Error ? error.message : 'chunk text unavailable'
        position()
      }
    }
  }
  const schedule = (target: HTMLElement, delay: number) => {
    window.clearTimeout(timer)
    window.clearTimeout(closeTimer)
    pending = target
    timer = window.setTimeout(() => void show(target), delay)
  }
  const leave = () => {
    window.clearTimeout(timer)
    pending = null
    closeTimer = window.setTimeout(() => {
      if (
        !popup.matches(':hover') &&
        !popup.contains(document.activeElement) &&
        !anchor?.matches(':focus-visible, :hover')
      )
        close()
    }, 180)
  }
  rt.root.addEventListener(
    'pointerover',
    event => {
      if (event.pointerType === 'touch') return
      const target =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('[data-lat-chunk]')
          : null
      if (target && target !== anchor) schedule(target, 160)
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener(
    'focusin',
    event => {
      const target =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('[data-lat-chunk]')
          : null
      if (target) schedule(target, 0)
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener(
    'pointerout',
    event => {
      if (event.target instanceof Element && event.target.closest('[data-lat-chunk]')) leave()
    },
    { signal: rt.signal },
  )
  popup.addEventListener('pointerenter', () => window.clearTimeout(closeTimer), {
    signal: rt.signal,
  })
  popup.addEventListener('pointerleave', leave, { signal: rt.signal })
  rt.root.addEventListener('focusout', leave, { signal: rt.signal })
  rt.root.addEventListener(
    'click',
    event => {
      if (event.target instanceof Element && event.target.closest('[data-lat-focus]')) close()
    },
    { signal: rt.signal },
  )
  rt.root.addEventListener('latent:links-render', close, { signal: rt.signal })
  document.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Escape') close()
    },
    { signal: rt.signal },
  )
  const viewportChanged = () => {
    const target = pending ?? anchor
    if (
      target?.isConnected &&
      (document.activeElement === target || popup.contains(document.activeElement))
    ) {
      const box = target.getBoundingClientRect()
      const pane = target.closest<HTMLElement>('.lat-links, .lat-panel')?.getBoundingClientRect()
      const left = Math.max(0, pane?.left ?? 0)
      const viewportWidth = document.documentElement.clientWidth
      const right = Math.min(viewportWidth, pane?.right ?? viewportWidth)
      const top = Math.max(0, pane?.top ?? 0)
      const bottom = Math.min(window.innerHeight, pane?.bottom ?? window.innerHeight)
      if (box.right > left && box.left < right && box.bottom > top && box.top < bottom) {
        if (anchor === target) position()
        return
      }
    }
    close()
  }
  rt.links.addEventListener('scroll', viewportChanged, { signal: rt.signal, passive: true })
  window.addEventListener('resize', viewportChanged, { signal: rt.signal })
  window.addEventListener('scroll', viewportChanged, { signal: rt.signal, passive: true })
  const observer = new ResizeObserver(position)
  observer.observe(popup)
  rt.signal.addEventListener(
    'abort',
    () => {
      close()
      observer.disconnect()
      popup.remove()
    },
    { once: true },
  )
}
