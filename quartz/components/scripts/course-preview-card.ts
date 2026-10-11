function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function text(root: ParentNode, selector: string): string {
  return root.querySelector(selector)?.textContent?.trim() ?? ''
}

function studySummary(root: ParentNode, selector: string): string {
  // Due and last depend on local review state; fetched pages provide notes and card counts.
  return text(root, selector).split(' · due')[0].trim()
}

function link(href: string, className: string, title: string): HTMLAnchorElement {
  const anchor = el('a', `${className} internal`, title)
  anchor.href = href
  anchor.dataset.noPopover = 'true'
  return anchor
}

function coursePath(url: URL): string {
  return url.pathname.replace(/\/index\/?$/, '').replace(/\/+$/, '')
}

export function coursePreviewTarget(href: URL, current: URL): URL | null {
  const path = coursePath(href)
  // every same-origin page under /courses, except the review app, raw PDFs and the page itself
  if (
    href.origin !== current.origin ||
    !/^\/courses(?:\/|$)/.test(path) ||
    path === '/courses/review' ||
    /\.pdf$/i.test(path) ||
    path === coursePath(current)
  ) {
    return null
  }
  const target = new URL(href)
  target.hash = ''
  target.search = ''
  return target
}

const MAX_ROWS = 5
const MAX_ROW_CHARS = 140

/** Cells of a table row (or children of a list item) joined with a middle dot. */
function rowText(row: Element): string {
  const parts = Array.from(row.children)
    .map(cell => (cell.textContent ?? '').replace(/\s+/g, ' ').trim())
    .filter(Boolean)
  const text =
    parts.length > 1 ? parts.join(' · ') : (row.textContent ?? '').replace(/\s+/g, ' ').trim()
  return text.length > MAX_ROW_CHARS ? `${text.slice(0, MAX_ROW_CHARS - 1)}…` : text
}

/** Rows of a `[data-course-rows]` container: the first table's body rows, else the first list's items. */
function cardRows(page: Document): Element[] {
  const container = page.querySelector('[data-course-rows]')
  if (!container) return []
  const table = container instanceof HTMLTableElement ? container : container.querySelector('table')
  const list = table ? null : container.querySelector('ul, ol')
  const rows = table
    ? Array.from(table.querySelectorAll('tbody > tr:not(.course-part-head)'))
    : list
      ? Array.from(list.children)
      : Array.from(container.children)
  return rows.filter(row => rowText(row).length > 0).slice(0, MAX_ROWS)
}

/** Crumb segments are separated by a bare slash in the strip; the card spaces them. */
function crumbText(head: Element): string {
  // the `courses` link opens every crumb on the page; the card drops it
  return text(head, '.course-crumb')
    .replace(/\s*\/\s*/g, ' / ')
    .replace(/^courses \/ /, '')
}

/**
 * Card for any course page that is not the home or the index: the `[data-course-card]` head
 * (crumb, title, meta), the excerpt, up to five rows, and an `open` link.
 */
function renderCourseCard(page: Document, href: URL, card: HTMLElement): void {
  const head = page.querySelector<HTMLElement>('[data-course-card]')
  const header = el('div', 'course-popover-header')
  const crumb = head ? crumbText(head) : ''
  if (crumb) header.appendChild(el('p', 'course-popover-crumb course-num', crumb))
  const title = el('h2', 'course-popover-title')
  const titleText =
    (head ? text(head, '.course-card-title') : '') ||
    text(page, 'h1') ||
    page.title ||
    href.pathname
  title.appendChild(link(href.toString(), 'course-popover-title-link', titleText))
  header.appendChild(title)
  // collection heads carry no meta; their body heading holds the member summary
  const meta =
    (head ? studySummary(head, '.course-count') : '') ||
    text(page, '.course-resource-heading .course-count')
  if (meta) header.appendChild(el('p', 'course-popover-meta', meta))
  card.appendChild(header)

  // reader and page heads already carry the description or the title in their meta line
  const excerpt =
    page.querySelector<HTMLMetaElement>('meta[name="description"]')?.content?.trim() ||
    text(page, '.course-transcript-list p')
  const repeats = excerpt === meta || excerpt.toLowerCase().includes(titleText.toLowerCase())
  if (excerpt && !repeats) card.appendChild(el('p', 'course-popover-description', excerpt))

  const rows = cardRows(page)
  if (rows.length > 0) {
    const list = el('ol', 'course-popover-list course-popover-rows')
    for (const row of rows) {
      const item = el('li', 'course-popover-item')
      const first = row.querySelector<HTMLAnchorElement>('a[href]')
      const entry = first
        ? link(first.href, 'course-popover-row', '')
        : el('div', 'course-popover-row')
      entry.appendChild(el('span', 'course-popover-topic', rowText(row)))
      item.appendChild(entry)
      list.appendChild(item)
    }
    card.appendChild(list)
  }
  card.appendChild(link(href.toString(), 'course-popover-more', 'open'))
}

export function renderCoursePreview(page: Document, href: URL, popoverInner: HTMLDivElement): void {
  popoverInner.dataset.contentType = 'text/x-course'
  const card = el('article', 'course-popover-card')
  const header = el('div', 'course-popover-header')
  const title = el('h2', 'course-popover-title')
  const home = page.querySelector<HTMLElement>('.course-home')
  const index = home?.querySelector('.course-index')

  if (!home) {
    renderCourseCard(page, href, card)
  } else if (index) {
    title.appendChild(link(href.toString(), 'course-popover-title-link', 'courses'))
    header.append(
      title,
      el(
        'p',
        'course-popover-meta',
        [text(index, '.course-bar h2'), studySummary(index, '.course-count')]
          .filter(Boolean)
          .join(' · '),
      ),
    )
    const list = el('ol', 'course-popover-list')
    for (const row of index.querySelectorAll<HTMLAnchorElement>('.course-row-link')) {
      const item = el('li', 'course-popover-item')
      const entry = link(row.href, 'course-popover-row', '')
      entry.append(
        el('span', 'course-num', text(row, '.course-num')),
        el('span', 'course-popover-topic', text(row, '.course-row-title')),
        el(
          'span',
          'course-popover-meta',
          [text(row, '.course-row-term'), studySummary(row, '.course-stats')]
            .filter(Boolean)
            .join(' · '),
        ),
      )
      item.appendChild(entry)
      list.appendChild(item)
    }
    card.append(header, list)
  } else {
    header.appendChild(
      el('p', 'course-popover-number course-num', text(home, '.course-masthead-num')),
    )
    title.appendChild(
      link(href.toString(), 'course-popover-title-link', text(home, '.course-masthead-title')),
    )
    const metadata = Array.from(home.querySelectorAll('.course-masthead-meta > span'))
      .filter(span => !span.classList.contains('course-sep'))
      .map(span => span.textContent?.trim())
      .filter(Boolean)
      .join(' · ')
    header.append(title, el('p', 'course-popover-meta', metadata))
    card.appendChild(header)
    const description = text(home, '.course-masthead-desc')
    if (description) card.appendChild(el('p', 'course-popover-description', description))
    card.appendChild(
      el('p', 'course-popover-summary', studySummary(home, '.course-main .course-count')),
    )

    const kind = text(home, '.course-main .course-bar h2')
    const rows = Array.from(
      home.querySelectorAll('.course-part:not(.course-part--psets) .course-unit-row'),
    )
    if (rows.length > 0) {
      card.appendChild(el('h3', 'course-popover-label', kind))
      const list = el('ol', 'course-popover-list course-popover-schedule')
      for (const row of rows.slice(0, 4)) {
        const item = el('li', 'course-popover-item')
        const topic = row.querySelector<HTMLAnchorElement>('a.course-unit-title')
        const entry = topic
          ? link(topic.href, 'course-popover-row', '')
          : el('div', 'course-popover-row')
        entry.append(
          el('span', 'course-num', text(row, '.course-c-n')),
          el('span', 'course-popover-topic', text(row, '.course-unit-title')),
        )
        item.appendChild(entry)
        list.appendChild(item)
      }
      card.append(
        list,
        link(`${href}#course-spine-h`, 'course-popover-more', `all ${rows.length} ${kind}`),
      )
    }
  }

  popoverInner.appendChild(card)
}
