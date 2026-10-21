export interface PagePreviewRow {
  title: string
  href?: string
  detail?: string
}

export interface PagePreview {
  kind: string
  title: string
  href: string
  description?: string
  meta?: string[]
  rows?: PagePreviewRow[]
}

export function previewText(value: string, limit: number): string {
  const text = value.replace(/\s+/g, ' ').trim()
  const points = Array.from(text)
  return points.length > limit
    ? `${points
        .slice(0, limit - 1)
        .join('')
        .trimEnd()}…`
    : text
}

export function previewHref(value: string, reference = window.location.href): string | undefined {
  try {
    const url = new URL(value, reference)
    return /^(https?:)$/.test(url.protocol) && !url.username && !url.password ? url.href : undefined
  } catch {
    return undefined
  }
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  node.className = className
  if (text) node.textContent = text
  return node
}

function link(href: string, title: string): HTMLAnchorElement {
  const anchor = el('a', 'page-preview-link', title)
  anchor.href = href
  anchor.dataset.noPopover = 'true'
  if (new URL(href).origin === window.location.origin) anchor.classList.add('internal')
  else {
    anchor.target = '_blank'
    anchor.rel = 'noopener noreferrer'
  }
  return anchor
}

/** Compact, inert content shared by custom layouts and saved-link previews. */
export function renderPagePreview(preview: PagePreview, inner: HTMLDivElement): void {
  inner.dataset.contentType = 'text/x-page-preview'
  inner.closest('.popover')?.classList.add('page-preview-popover')
  const card = el('article', 'page-preview-card')
  card.dataset.kind = preview.kind
  const title = el('h2', 'page-preview-title')
  title.appendChild(link(preview.href, preview.title))
  card.appendChild(title)
  if (preview.meta?.length) {
    const meta = el('ul', 'page-preview-meta')
    for (const value of preview.meta) meta.appendChild(el('li', '', value))
    card.appendChild(meta)
  }
  if (preview.description) {
    card.appendChild(el('p', 'page-preview-description', preview.description))
  }
  if (preview.rows?.length) {
    const list = el('ul', 'page-preview-list')
    for (const row of preview.rows.slice(0, 4)) {
      const item = el('li', 'page-preview-row')
      item.appendChild(row.href ? link(row.href, row.title) : el('span', '', row.title))
      if (row.detail) item.appendChild(el('span', 'page-preview-detail', row.detail))
      list.appendChild(item)
    }
    card.appendChild(list)
  }
  inner.appendChild(card)
}
