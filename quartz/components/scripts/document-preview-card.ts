import type {
  BasePreview,
  BasePreviewSample,
  CanvasPreview,
  DocumentPreview,
} from '../../util/document-preview'

const SVG_NS = 'http://www.w3.org/2000/svg'
// The frame is a 16:10 window; the miniature's box is widened to match so labels can sit in percentages.
const FRAME_ASPECT = 16 / 10
// Labels need roughly 2.5rem of a ~20rem frame before they read as text.
const LABEL_MIN_FRACTION = 0.12

// Lucide paths: layout-dashboard for canvas, the base toolbar's view icons for bases.
const ICONS: Record<string, string[]> = {
  canvas: [
    'M4 3h5a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z',
    'M15 3h5a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z',
    'M15 12h5a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1z',
    'M4 16h5a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1z',
  ],
  table: [
    'M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z',
    'M3 9h18',
    'M3 15h18',
    'M9 3v18',
    'M15 3v18',
  ],
  list: ['M8 6h13', 'M8 12h13', 'M8 18h13', 'M3 6h.01', 'M3 12h.01', 'M3 18h.01'],
  cards: [
    'M4 3h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z',
    'M15 3h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z',
    'M15 14h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z',
    'M4 14h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z',
  ],
  map: [
    'M12 2a10 10 0 1 0 0 20 10 10 0 1 0 0-20z',
    'M12 2c-2.2 2.7-4 6.2-4 10s1.8 7.3 4 10c2.2-2.7 4-6.2 4-10s-1.8-7.3-4-10z',
    'M2 12h20',
  ],
}

const VIEW_ICONS: Record<string, string> = {
  table: 'table',
  board: 'table',
  calendar: 'list',
  list: 'list',
  card: 'cards',
  cards: 'cards',
  gallery: 'cards',
  map: 'map',
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag)
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value))
  return node
}

function icon(name: string): SVGSVGElement {
  const glyph = svg('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' })
  for (const d of ICONS[name] ?? ICONS.table) glyph.appendChild(svg('path', { d }))
  return glyph
}

const plural = (count: number, one: string, many = `${one}s`) =>
  `${count} ${count === 1 ? one : many}`

/** Presets are styled by data attribute; a hex colour rides a custom property. */
function paint(node: Element & ElementCSSInlineStyle, color: string | null) {
  if (!color) return
  if (color.startsWith('#')) node.style.setProperty('--document-preview-accent', color)
  else node.setAttribute('data-color', color)
}

function framedBox({ box }: CanvasPreview) {
  let { x, y, w, h } = box
  if (w / h > FRAME_ASPECT) {
    const height = w / FRAME_ASPECT
    y -= (height - h) / 2
    h = height
  } else {
    const width = h * FRAME_ASPECT
    x -= (width - w) / 2
    w = width
  }
  return { x, y, w, h }
}

function canvasMiniature(preview: CanvasPreview): HTMLElement | null {
  if (preview.nodes.length === 0 || preview.box.w <= 0 || preview.box.h <= 0) return null
  const box = framedBox(preview)
  const figure = el('div', 'document-popover-canvas')
  const drawing = svg('svg', {
    viewBox: `${box.x} ${box.y} ${box.w} ${box.h}`,
    preserveAspectRatio: 'none',
  })
  const edges = svg('g', { class: 'document-popover-edges' })
  for (const edge of preview.edges) {
    const path = svg('path', { d: edge.d, 'vector-effect': 'non-scaling-stroke' })
    paint(path, edge.color)
    edges.appendChild(path)
  }
  // Groups, then edges, then cards: lines run under the cards they join.
  const groups = svg('g', { class: 'document-popover-groups' })
  const nodes = svg('g', { class: 'document-popover-nodes' })
  const labels = el('div', 'document-popover-labels')
  for (const node of preview.nodes) {
    const rect = svg('rect', {
      x: node.x,
      y: node.y,
      width: node.w,
      height: node.h,
      'data-kind': node.kind,
      'vector-effect': 'non-scaling-stroke',
    })
    paint(rect, node.color)
    const layer = node.kind === 'group' ? groups : nodes
    layer.appendChild(rect)

    const width = node.w / box.w
    if (!node.label || width < LABEL_MIN_FRACTION) continue
    const label = el('span', `document-popover-label is-${node.kind}`, node.label)
    label.style.left = `${((node.x - box.x) / box.w) * 100}%`
    label.style.top = `${((node.y - box.y) / box.h) * 100}%`
    label.style.width = `${width * 100}%`
    labels.appendChild(label)
  }
  drawing.append(groups, edges, nodes)
  figure.append(drawing, labels)
  return figure
}

function baseMiniature(sample: BasePreviewSample): HTMLElement {
  switch (sample.type) {
    case 'cards': {
      const grid = el('ol', 'document-popover-cards')
      for (const item of sample.items) {
        const cell = el('li')
        if (item.image) {
          const img = el('img')
          img.alt = ''
          img.decoding = 'async'
          img.src = new URL(item.image, window.location.href).toString()
          cell.appendChild(img)
        } else cell.appendChild(el('span', undefined, item.title))
        grid.appendChild(cell)
      }
      return grid
    }
    case 'table': {
      const table = el('table', 'document-popover-table')
      if (sample.columns.length > 0) {
        const head = el('tr')
        for (const column of sample.columns) head.appendChild(el('th', undefined, column))
        table.appendChild(el('thead')).appendChild(head)
      }
      const body = table.appendChild(el('tbody'))
      for (const row of sample.rows) {
        const line = el('tr')
        for (const cell of row) line.appendChild(el('td', undefined, cell))
        body.appendChild(line)
      }
      return table
    }
    case 'list': {
      const list = el('ul', 'document-popover-list')
      for (const item of sample.items) list.appendChild(el('li', undefined, item))
      return list
    }
  }
}

const canvasMeta = ({ counts }: CanvasPreview): string[] => [
  'canvas',
  plural(counts.cards, 'card'),
  ...(counts.groups > 0 ? [plural(counts.groups, 'group')] : []),
  ...(counts.connections > 0 ? [plural(counts.connections, 'connection')] : []),
]

function baseMeta({ views }: BasePreview): string[] {
  const [first] = views
  if (!first) return ['base']
  return [
    'base',
    first.name,
    plural(first.count, 'item'),
    ...(views.length > 1 ? [plural(views.length, 'view')] : []),
  ]
}

export function renderDocumentPreview(
  preview: DocumentPreview,
  href: string,
  popoverInner: HTMLElement,
) {
  popoverInner.dataset.contentType = `text/x-${preview.type}`

  const card = el('article', 'document-popover-card')
  card.dataset.type = preview.type

  const link = el('a', 'document-popover-window')
  link.href = href
  const frame = el('div', 'document-popover-frame')
  frame.ariaHidden = 'true'
  const miniature =
    preview.type === 'canvas'
      ? canvasMiniature(preview)
      : preview.sample
        ? baseMiniature(preview.sample)
        : null
  if (miniature) frame.appendChild(miniature)
  else frame.classList.add('is-empty')

  const badge = el('span', 'document-popover-badge')
  badge.appendChild(
    icon(
      preview.type === 'canvas' ? 'canvas' : (VIEW_ICONS[preview.views[0]?.type ?? ''] ?? 'table'),
    ),
  )
  link.append(frame, badge, el('span', 'document-popover-title', preview.title))
  card.appendChild(link)

  const meta = el('ul', 'document-popover-meta')
  for (const item of preview.type === 'canvas' ? canvasMeta(preview) : baseMeta(preview))
    meta.appendChild(el('li', undefined, item))
  card.appendChild(meta)

  if (preview.type === 'base' && preview.description)
    card.appendChild(el('p', 'document-popover-description', preview.description))

  popoverInner.appendChild(card)
}
