import type { PressKitPage } from '../PressKit'
import { buildSelect, type Select } from '../controls/select'
import { currentNavSignal } from './nav-lifecycle'
import {
  FORMATS,
  cardFaces,
  palettes,
  renderCard,
  type CardModel,
  type FormatKey,
  type Fonts,
  type SiteColors,
  type ThemeKey,
} from './press-kit-card'
import {
  runsFromNode,
  runsFromRange,
  runsFromText,
  runsToPlain,
  runsToText,
  type Run,
} from './press-kit-text'

export type PressKitTab = 'note' | 'quote' | 'links'

const TABS: PressKitTab[] = ['note', 'quote', 'links']
const FORMAT_KEYS = Object.keys(FORMATS) as FormatKey[]
const THEME_KEYS: ThemeKey[] = ['papier', 'encre', 'rose', 'sauge']
const PREFS_KEY = 'press-kit:prefs'
const CHORD_WINDOW_MS = 1000
const X_LIMIT = 280
// X shortens every link to a t.co URL of this length.
const X_LINK_LENGTH = 23
const SOURCES = 'article.popover-hint, .stream-entry-body'

interface Subject {
  kind: 'page' | 'entry'
  url: string
  title: Run[]
  description: Run[]
  excerpt: Run[]
  date?: string
  iso?: string
  reading?: string
  tags: string[]
  crumbs: string[]
  /** headings that a link can point into */
  headings: HTMLElement[]
}

interface Quote {
  runs: Run[]
  /** false while the quote is the page's own excerpt, before any selection or edit */
  chosen: boolean
  /** `text=` directive for a scroll-to-text link, when the passage is plain prose */
  fragment?: string
  section?: HTMLElement
}

interface OpenRequest {
  tab?: PressKitTab
  entry?: HTMLElement
  entryUrl?: string
  range?: Range
}

let dialog: HTMLDialogElement | null = null
let page: PressKitPage | null = null
let activeSignal: AbortSignal | undefined

let tab: PressKitTab = 'note'
let subject: Subject | null = null
let quote: Quote = { runs: [], chosen: false }
let anchor = 'page'
let format: FormatKey = 'portrait'
let theme: ThemeKey = 'papier'
const show = { description: true, excerpt: false, meta: true, highlight: false, section: true }
let noteEdited = false
let anchorSelect: Select<string> | null = null
let unmountAnchor: (() => void) | null = null
let fonts: Fonts | null = null
let avatar: HTMLImageElement | undefined
let frame = 0
let blob: Promise<Blob | null> | null = null
let blobTimer = 0
let pendingG = -Infinity
let lastRange: { range: Range; at: number } | null = null
// A selection survives this long after it collapses, so a button click elsewhere can still quote it.
const RANGE_TTL_MS = 60_000
let pill: HTMLButtonElement | null = null

const $ = <T extends Element = HTMLElement>(selector: string) =>
  dialog?.querySelector<T>(selector) ?? null

function toast(message: string) {
  document.dispatchEvent(new CustomEvent('toast', { detail: { message } }))
}

function readPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFS_KEY) ?? 'null') as {
      format?: FormatKey
      theme?: ThemeKey
    } | null
    if (saved?.format && FORMAT_KEYS.includes(saved.format)) format = saved.format
    if (saved?.theme && THEME_KEYS.includes(saved.theme)) {
      theme = saved.theme
      return
    }
  } catch {}
  theme = document.documentElement.getAttribute('saved-theme') === 'dark' ? 'encre' : 'papier'
}

function writePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ format, theme }))
  } catch {}
}

function resolveFonts(): Fonts {
  const root = getComputedStyle(document.documentElement)
  const variable = (name: string, fallback: string) =>
    root.getPropertyValue(name).trim() || fallback
  const heading = document.querySelector('h1.article-title')
  const weight = heading ? Number.parseInt(getComputedStyle(heading).fontWeight, 10) : 700
  return {
    header: variable('--headerFont', 'system-ui, sans-serif'),
    headerWeight: Number.isFinite(weight) ? weight : 700,
    body: variable('--bodyFont', 'system-ui, sans-serif'),
    serif: '"Parclo Serif", Cambria, Georgia, serif',
    mono: variable('--codeFont', 'ui-monospace, monospace'),
  }
}

/** Draws at once with whatever is loaded, then again once every card face has arrived. */
function loadFonts(target: Fonts) {
  const loads = cardFaces(target).map(face => document.fonts.load(face, 'Aa“…').catch(() => []))
  const timeout = new Promise(resolve => setTimeout(resolve, 3000))
  void Promise.race([Promise.all(loads), timeout]).then(() => {
    if (fonts === target) schedule()
  })
}

function loadAvatar() {
  if (avatar) return
  const img = new Image()
  img.decoding = 'async'
  img.src = '/static/icon.webp'
  img.onload = () => {
    avatar = img
    if (subject?.kind === 'entry') schedule()
  }
}

function absolute(href: string): string {
  try {
    const url = new URL(href, page!.url)
    // A local preview still shares the canonical address.
    if (url.origin === location.origin) return `https://${page!.site}${url.pathname}${url.hash}`
    return url.href
  } catch {
    return href
  }
}

const display = (url: string) => url.replace(/^https?:\/\//, '').replace(/\/$/, '')

function headingText(heading: HTMLElement): string {
  const alias = heading.getAttribute('data-heading-alias')?.trim()
  if (alias) return alias
  return runsToPlain(runsFromNode(heading)).replace(/\s+/g, ' ').trim()
}

function excerptOf(root: Element | null): Run[] {
  if (!root) return []
  for (const p of Array.from(root.querySelectorAll(':scope > p, :scope > div > p'))) {
    if (p.closest('.callout, figure, table, .footnotes, details')) continue
    const runs = runsFromNode(p)
    if (runsToPlain(runs).length >= 60) return runs
  }
  return []
}

function pageSubject(): Subject {
  const title = document.querySelector('[data-article-title] .article-title')
  const description = document.querySelector('[data-article-title] .description')
  const article = document.querySelector('article.popover-hint')
  return {
    kind: 'page',
    url: page!.url,
    title: title ? runsFromNode(title) : [{ text: document.title, style: 'roman' }],
    description: description ? runsFromNode(description) : [],
    excerpt: excerptOf(article),
    date: page!.date,
    iso: page!.iso,
    reading: page!.reading,
    tags: page!.tags,
    crumbs: page!.crumbs,
    headings: Array.from(
      document.querySelectorAll<HTMLElement>('.page-content :is(h2, h3, h4)[id]'),
    ).filter(heading => !heading.closest('.press-kit, .stream-entry') && headingText(heading)),
  }
}

function entrySubject(entry: HTMLElement, href: string | undefined): Subject {
  const title = entry.querySelector('.stream-entry-title')
  const body = entry.querySelector('.stream-entry-body')
  const content = document.createDocumentFragment()
  body
    ?.querySelectorAll(':scope > .stream-entry-description, :scope > .stream-entry-content')
    .forEach(node => content.append(node.cloneNode(true)))
  const time = entry.querySelector<HTMLTimeElement>(
    '.stream-entry-date time, time.stream-entry-date',
  )
  const url = absolute(href ?? `${location.pathname}#${entry.id}`)
  return {
    kind: 'entry',
    url,
    title: title ? runsFromNode(title) : [],
    description: [],
    excerpt: runsFromNode(content),
    date: time?.textContent?.trim() || undefined,
    iso: time?.dateTime || undefined,
    tags: Array.from(entry.querySelectorAll('.stream-entry-tag'), tag => tag.textContent!.trim())
      .filter(Boolean)
      .slice(0, 4),
    crumbs: [page!.site, ...new URL(url).pathname.split('/').filter(Boolean).slice(0, -1)],
    headings: [],
  }
}

function sourceOf(node: Node): Element | null {
  const el = node instanceof Element ? node : node.parentElement
  const source = el?.closest(SOURCES) ?? null
  return source && !source.closest('.press-kit') ? source : null
}

function selectedRange(): Range | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null
  const range = selection.getRangeAt(0)
  if (!sourceOf(range.commonAncestorContainer)) return null
  return selection.toString().trim().length >= 3 ? range.cloneRange() : null
}

function sectionBefore(node: Node, headings: HTMLElement[]): HTMLElement | undefined {
  let found: HTMLElement | undefined
  for (const heading of headings) {
    if (heading === node || heading.contains(node)) return heading
    if (heading.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING) found = heading
    else break
  }
  return found
}

/** Scroll-to-text directive; ranges through math skip it, since KaTeX text never matches. */
function fragmentFor(range: Range): string | undefined {
  const fragment = range.cloneContents()
  if (fragment.querySelector('.katex')) return undefined
  const words = range.toString().replace(/\s+/g, ' ').trim().split(' ').filter(Boolean)
  if (words.length === 0) return undefined
  const encode = (text: string) =>
    encodeURIComponent(text).replace(/-/g, '%2D').replace(/,/g, '%2C').replace(/&/g, '%26')
  if (words.length <= 8) return `text=${encode(words.join(' '))}`
  return `text=${encode(words.slice(0, 4).join(' '))},${encode(words.slice(-4).join(' '))}`
}

function quoteFrom(range: Range | null): Quote {
  if (!range || !subject) return { runs: subject?.excerpt ?? [], chosen: false }
  const source = sourceOf(range.commonAncestorContainer)
  return {
    runs: runsFromRange(range, source),
    chosen: true,
    fragment: fragmentFor(range),
    section: sectionBefore(range.startContainer, subject.headings),
  }
}

function sectionForCard(): HTMLElement | undefined {
  if (anchor.startsWith('#')) {
    return subject?.headings.find(heading => `#${heading.id}` === anchor)
  }
  return quote.section
}

function shareUrl(): string {
  if (!subject) return page?.url ?? location.href
  const base = subject.url.split('#')[0]
  if (anchor === 'passage' && quote.fragment) {
    const section = quote.section ? `#${quote.section.id}` : '#'
    return `${base}${section}:~:${quote.fragment}`
  }
  if (anchor.startsWith('#')) return `${base}${anchor}`
  return subject.url
}

function cardUrl(): string {
  if (!subject) return ''
  const section = tab === 'quote' && show.section ? sectionForCard() : undefined
  return display(section ? `${subject.url.split('#')[0]}#${section.id}` : subject.url)
}

function model(): CardModel {
  const s = subject!
  const section = sectionForCard()
  const kind = tab === 'quote' ? 'quote' : s.kind === 'entry' ? 'entry' : 'note'
  return {
    kind,
    crumbs: s.crumbs,
    title: s.title.length > 0 ? s.title : [{ text: page!.site, style: 'roman' }],
    description: s.description,
    body: tab === 'quote' ? quote.runs : s.excerpt,
    date: s.date,
    reading: s.reading,
    tags: s.tags,
    url: cardUrl(),
    section: section ? headingText(section) : undefined,
    author: kind === 'entry' ? page!.handle : page!.author,
    avatar,
    show: { ...show, excerpt: kind === 'entry' || show.excerpt },
  }
}

function caption() {
  const spec = FORMATS[format]
  const el = $('.press-kit-caption')
  if (!el) return
  el.textContent =
    tab === 'links'
      ? 'aperçu du lien partagé'
      : `${spec.width} × ${spec.height} · png · ${spec.label}`
}

function draw() {
  frame = 0
  if (!dialog?.open || !subject || !fonts || !page) return
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  const unfurl = $('.press-kit-unfurl')
  if (!canvas || !unfurl) return
  dialog.dataset.format = format
  caption()
  if (tab === 'links') {
    canvas.hidden = true
    unfurl.hidden = false
    renderLinks()
    return
  }
  canvas.hidden = false
  unfurl.hidden = true
  const colors = page.colors as unknown as { light: SiteColors; dark: SiteColors }
  renderCard(canvas, model(), format, palettes(colors.light, colors.dark)[theme], fonts)
  canvas.setAttribute('aria-label', `aperçu de la carte, ${FORMATS[format].label}, ${theme}`)
  window.clearTimeout(blobTimer)
  blob = null
  blobTimer = window.setTimeout(() => {
    blob = new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  }, 160)
}

function schedule() {
  if (!frame) frame = requestAnimationFrame(draw)
}

function currentBlob(): Promise<Blob | null> {
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  if (frame) {
    cancelAnimationFrame(frame)
    draw()
  }
  if (!blob && canvas) {
    window.clearTimeout(blobTimer)
    blob = new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  }
  return blob ?? Promise.resolve(null)
}

function fileName(): string {
  const tail = subject?.url.split('#')[0].split('/').filter(Boolean).at(-1) ?? page!.handle
  const name = tail.includes('.') && subject?.kind === 'page' ? page!.handle : tail
  return `${page!.handle}-${name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${tab}-${format}.png`
}

async function download() {
  const data = await currentBlob()
  if (!data) return toast("impossible de générer l'image")
  const url = URL.createObjectURL(data)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName()
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

async function copyImage() {
  if (typeof ClipboardItem === 'undefined' || !navigator.clipboard?.write) {
    return toast('copie d’image non prise en charge ici')
  }
  // Safari keeps the user gesture only when the item is built synchronously around a promise.
  const data = currentBlob().then(value => {
    if (!value) throw new Error('empty image')
    return value
  })
  try {
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': data })])
    toast('image copiée')
  } catch {
    toast("impossible de copier l'image")
  }
}

function canShareFiles(): boolean {
  try {
    const probe = new File([new Uint8Array(1)], 'probe.png', { type: 'image/png' })
    return !!navigator.canShare?.({ files: [probe] })
  } catch {
    return false
  }
}

async function shareImage() {
  const data = await currentBlob()
  if (!data) return
  const file = new File([data], fileName(), { type: 'image/png' })
  try {
    await navigator.share({ files: [file], title: titleText(), url: shareUrl() })
  } catch (error) {
    if ((error as DOMException)?.name !== 'AbortError') toast('partage impossible')
  }
}

const titleText = () => runsToPlain(subject?.title ?? []) || page?.site || ''

function defaultNote(): string {
  if (!subject) return ''
  const title = titleText()
  const passage = anchor === 'passage' || tab === 'quote' ? runsToPlain(quote.runs) : ''
  if (passage && quote.chosen) {
    const clipped = passage.length > 180 ? `${passage.slice(0, 177).trimEnd()}…` : passage
    return `“${clipped}” — ${title}`
  }
  const description = runsToPlain(subject.description)
  return description ? `${title} — ${description}` : title
}

function intentHref(intent: string, text: string, url: string): string {
  const e = encodeURIComponent
  const title = titleText()
  switch (intent) {
    case 'x':
      return `https://x.com/intent/post?text=${e(text)}&url=${e(url)}`
    case 'bluesky':
      return `https://bsky.app/intent/compose?text=${e(`${text} ${url}`.trim())}`
    case 'threads':
      return `https://www.threads.net/intent/post?text=${e(`${text} ${url}`.trim())}`
    case 'linkedin':
      return `https://www.linkedin.com/sharing/share-offsite/?url=${e(url)}`
    case 'hn':
      return `https://news.ycombinator.com/submitlink?u=${e(url)}&t=${e(title)}`
    case 'email':
      return `mailto:?subject=${e(title)}&body=${e(`${text}\n\n${url}`)}`
    default:
      return url
  }
}

function bibtex(url: string): string {
  const year = (subject?.iso ?? new Date().toISOString()).slice(0, 4)
  const family = page!.author.split(' ').at(-1)!.toLowerCase()
  const word =
    titleText()
      .toLowerCase()
      .match(/[a-z0-9]+/)?.[0] ?? 'note'
  const given = page!.author.split(' ').slice(0, -1).join(' ')
  return [
    `@misc{${family}${year}${word},`,
    `  author = {${page!.author.split(' ').at(-1)}, ${given}},`,
    `  title = {{${titleText().replace(/[{}]/g, '')}}},`,
    `  year = {${year}},`,
    `  howpublished = {\\url{${url}}},`,
    `  urldate = {${new Date().toISOString().slice(0, 10)}},`,
    `  note = {${page!.site}}`,
    `}`,
  ].join('\n')
}

function copyText(kind: string): string {
  const url = shareUrl()
  const title = titleText()
  const passage = anchor === 'passage' ? runsToText(quote.runs) : ''
  switch (kind) {
    case 'markdown':
      return passage
        ? `> ${passage.replace(/\n\n/g, '\n>\n> ')}\n>\n> — [${title}](${url})`
        : `[${title}](${url})`
    case 'wikilink': {
      const section = sectionForCard()
      const target = subject?.kind === 'entry' ? display(url) : page!.slug || 'index'
      const hash = anchor.startsWith('#') && section ? `#${headingText(section)}` : ''
      return `[[${target}${hash}|${title}]]`
    }
    case 'bibtex':
      return bibtex(url)
    case 'citation': {
      const family = page!.author.split(' ').at(-1)
      const given = page!.author.split(' ').slice(0, -1).join(' ')
      const date = subject?.date ? ` ${subject.date}.` : ''
      return `${family}, ${given}. “${title}.” ${page!.site},${date} ${url}.`
    }
    default:
      return url
  }
}

async function copy(kind: string, label: string) {
  try {
    await navigator.clipboard.writeText(copyText(kind))
    toast(`${label} copié`)
  } catch {
    toast('copie impossible')
  }
}

function anchorGroups() {
  const options: { value: string; label: string }[] = [{ value: 'page', label: 'page entière' }]
  if (quote.fragment) options.push({ value: 'passage', label: 'passage sélectionné' })
  const headings = (subject?.headings ?? []).map(heading => ({
    value: `#${heading.id}`,
    label: `${'  '.repeat(Math.max(0, Number(heading.tagName[1]) - 2))}${headingText(heading)}`,
  }))
  return [{ options }, ...(headings.length > 0 ? [{ label: 'sections', options: headings }] : [])]
}

function mountAnchor(signal: AbortSignal) {
  const host = $('.press-kit-anchor')
  if (!host) return
  unmountAnchor?.()
  host.replaceChildren()
  anchorSelect = buildSelect<string>({
    id: 'press-kit-anchor-menu',
    label: 'ancre',
    groups: anchorGroups(),
    selected: anchor,
    className: 'press-kit-select',
    onSelect: value => {
      anchor = value
      if (!noteEdited) $<HTMLTextAreaElement>('#press-kit-note')!.value = defaultNote()
      schedule()
    },
  })
  host.append(anchorSelect.element)
  const unmount = anchorSelect.mount()
  unmountAnchor = unmount
  signal.addEventListener('abort', unmount, { once: true })
}

function renderLinks() {
  const url = shareUrl()
  const input = $<HTMLInputElement>('#press-kit-url')
  if (input) input.value = url
  const note = $<HTMLTextAreaElement>('#press-kit-note')
  if (note && !noteEdited) note.value = defaultNote()
  const text = note?.value.trim() ?? ''
  dialog?.querySelectorAll<HTMLAnchorElement>('[data-intent]').forEach(link => {
    link.href = intentHref(link.dataset.intent!, text, url)
  })
  const count = $('#press-kit-note-count')
  if (count) {
    const used = text.length + (text ? 1 : 0) + X_LINK_LENGTH
    count.textContent = `${used} / ${X_LIMIT} sur x`
    count.toggleAttribute('data-over', used > X_LIMIT)
  }
  const image = $<HTMLImageElement>('.press-kit-unfurl-image')
  const og =
    subject?.kind === 'entry'
      ? null
      : document.querySelector<HTMLMetaElement>('meta[property="og:image"]')?.content
  if (image) {
    const src = og ?? '/static/og-image.webp'
    if (image.getAttribute('src') !== src) image.src = src
  }
  const title = $('.press-kit-unfurl-title')
  if (title) title.textContent = titleText()
  const site = $('.press-kit-unfurl-site')
  if (site) site.textContent = display(url).split('/')[0]
}

const HELP: Record<'image' | 'links', string> = {
  image: '{1}{2}{3} onglets · {f} format · {t} thème · {d} télécharger · {c} copier · {esc} fermer',
  links: '{1}{2}{3} onglets · {esc} fermer',
}

function setHelp() {
  const status = $('.press-kit-help')
  if (!status) return
  const template = HELP[tab === 'links' ? 'links' : 'image']
  status.replaceChildren(
    ...template.split(/\{([^}]+)\}/).map((part, index) => {
      if (index % 2 === 0) return part
      const kbd = document.createElement('kbd')
      kbd.textContent = part
      return kbd
    }),
  )
}

function syncControls() {
  if (!dialog) return
  dialog.dataset.tab = tab
  dialog.dataset.subject = subject?.kind ?? 'page'
  dialog.querySelectorAll<HTMLButtonElement>('.press-kit-tab').forEach(button => {
    const selected = button.dataset.tab === tab
    button.setAttribute('aria-selected', String(selected))
    button.tabIndex = selected ? 0 : -1
  })
  $('#press-kit-panel')?.setAttribute('aria-labelledby', `press-kit-tab-${tab}`)
  dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-format"]')
    .forEach(input => (input.checked = input.value === format))
  dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-theme"]')
    .forEach(input => (input.checked = input.value === theme))
  dialog.querySelectorAll<HTMLInputElement>('.press-kit-check input').forEach(input => {
    input.checked = show[input.name as keyof typeof show]
  })
  const textarea = $<HTMLTextAreaElement>('#press-kit-quote')
  if (textarea && document.activeElement !== textarea) textarea.value = runsToText(quote.runs)
  anchorSelect?.setSelected(anchor)
  setHelp()
}

function setTab(next: PressKitTab, focus = false) {
  tab = next
  syncControls()
  if (focus) $<HTMLButtonElement>(`#press-kit-tab-${next}`)?.focus()
  schedule()
}

function open(request: OpenRequest = {}) {
  if (!dialog || !page) return
  if (dialog.open) {
    if (request.tab) setTab(request.tab)
    return
  }
  readPrefs()
  fonts = resolveFonts()
  loadFonts(fonts)
  loadAvatar()

  subject = request.entry ? entrySubject(request.entry, request.entryUrl) : pageSubject()
  const recent =
    lastRange && performance.now() - lastRange.at < RANGE_TTL_MS ? lastRange.range : null
  const candidate = request.range ?? selectedRange() ?? recent
  // An entry card only quotes text from inside that entry.
  const usable =
    candidate &&
    sourceOf(candidate.commonAncestorContainer) &&
    (!request.entry || request.entry.contains(candidate.commonAncestorContainer))
      ? candidate
      : null
  quote = quoteFrom(usable)
  anchor = quote.fragment ? 'passage' : 'page'
  noteEdited = false
  const note = $<HTMLTextAreaElement>('#press-kit-note')
  if (note) note.value = ''
  tab = request.tab ?? (usable ? 'quote' : 'note')

  mountAnchor(currentNavSignal())
  const shareImageButton = $<HTMLButtonElement>('[data-action="share-image"]')
  if (shareImageButton) shareImageButton.hidden = !canShareFiles()
  const shareLinkButton = $<HTMLButtonElement>('[data-action="share-link"]')
  if (shareLinkButton) shareLinkButton.hidden = typeof navigator.share !== 'function'

  hidePill()
  syncControls()
  dialog.showModal()
  $<HTMLButtonElement>(`#press-kit-tab-${tab}`)?.focus()
  draw()
}

function cycle<T>(list: T[], value: T, delta: number): T {
  return list[(list.indexOf(value) + delta + list.length) % list.length]
}

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false
  return (
    target.matches('input:not([type="radio"]):not([type="checkbox"]), textarea, select') ||
    (target as HTMLElement).isContentEditable ||
    target.closest('[role="listbox"]') !== null
  )
}

function onDialogKeyDown(event: KeyboardEvent) {
  if (event.isComposing || event.metaKey || event.ctrlKey || event.altKey) return
  const { key } = event
  if (event.target instanceof Element && event.target.matches('.press-kit-tab')) {
    if (key === 'ArrowRight' || key === 'ArrowLeft') {
      setTab(cycle(TABS, tab, key === 'ArrowRight' ? 1 : -1), true)
      event.preventDefault()
      return
    }
    if (key === 'Home' || key === 'End') {
      setTab(key === 'Home' ? TABS[0] : TABS.at(-1)!, true)
      event.preventDefault()
      return
    }
  }
  if (isEditable(event.target)) return
  const imageTab = tab !== 'links'
  if (key === '1' || key === '2' || key === '3') {
    setTab(TABS[Number(key) - 1], true)
  } else if (imageTab && (key === 'f' || key === 'F')) {
    format = cycle(FORMAT_KEYS, format, key === 'F' ? -1 : 1)
    writePrefs()
    syncControls()
    schedule()
  } else if (imageTab && (key === 't' || key === 'T')) {
    theme = cycle(THEME_KEYS, theme, key === 'T' ? -1 : 1)
    writePrefs()
    syncControls()
    schedule()
  } else if (imageTab && key === 'd') {
    void download()
  } else if (imageTab && key === 'c') {
    void copyImage()
  } else {
    return
  }
  event.preventDefault()
  event.stopPropagation()
}

function onDocumentKeyDown(event: KeyboardEvent) {
  if (dialog?.open) return
  if (event.ctrlKey || event.metaKey || event.altKey || isEditable(event.target)) {
    pendingG = -Infinity
    return
  }
  if (event.key === 's' && event.timeStamp - pendingG < CHORD_WINDOW_MS) {
    event.preventDefault()
    pendingG = -Infinity
    open()
    return
  }
  pendingG = event.key === 'g' ? event.timeStamp : -Infinity
}

function hidePill() {
  pill?.remove()
  pill = null
}

/** A small "citer" button under a fresh selection, as on Substack and Medium. */
function showPill(range: Range) {
  const rects = range.getClientRects()
  const rect = rects[rects.length - 1] ?? range.getBoundingClientRect()
  if (!rect || (rect.width === 0 && rect.height === 0)) return hidePill()
  if (!pill) {
    pill = document.createElement('button')
    pill.type = 'button'
    pill.className = 'press-kit-pill'
    pill.textContent = 'citer'
    pill.setAttribute('aria-label', 'citer la sélection en image')
    // Keeps the selection alive through the click.
    pill.addEventListener('pointerdown', event => event.preventDefault())
    pill.addEventListener('click', () => open({ tab: 'quote', range: lastRange?.range }))
    document.body.append(pill)
  }
  const width = pill.offsetWidth || 48
  const left = Math.min(
    Math.max(8, rect.right + window.scrollX - width / 2),
    window.scrollX + document.documentElement.clientWidth - width - 8,
  )
  pill.style.left = `${left}px`
  pill.style.top = `${rect.bottom + window.scrollY + 8}px`
}

function onSelectionSettled() {
  if (dialog?.open) return
  const range = selectedRange()
  if (!range) {
    hidePill()
    return
  }
  lastRange = { range, at: performance.now() }
  showPill(range)
}

function bindDialog(signal: AbortSignal) {
  const modal = dialog!
  modal.addEventListener('keydown', onDialogKeyDown, { signal })
  modal.addEventListener(
    'close',
    () => {
      window.clearTimeout(blobTimer)
      if (frame) cancelAnimationFrame(frame)
      frame = 0
    },
    { signal },
  )
  $('.press-kit-close')?.addEventListener('click', () => modal.close(), { signal })
  modal.addEventListener(
    'click',
    event => {
      if (event.target !== modal) return
      const rect = modal.getBoundingClientRect()
      const inside =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom
      if (!inside) modal.close()
    },
    { signal },
  )
  modal
    .querySelectorAll<HTMLButtonElement>('.press-kit-tab')
    .forEach(button =>
      button.addEventListener('click', () => setTab(button.dataset.tab as PressKitTab), { signal }),
    )
  modal.addEventListener(
    'change',
    event => {
      const input = event.target
      if (!(input instanceof HTMLInputElement)) return
      if (input.name === 'press-kit-format') format = input.value as FormatKey
      else if (input.name === 'press-kit-theme') theme = input.value as ThemeKey
      else if (input.type === 'checkbox' && input.name in show) {
        show[input.name as keyof typeof show] = input.checked
      } else return
      writePrefs()
      schedule()
    },
    { signal },
  )
  $<HTMLTextAreaElement>('#press-kit-quote')?.addEventListener(
    'input',
    event => {
      const value = (event.target as HTMLTextAreaElement).value
      quote = { ...quote, runs: runsFromText(value), chosen: true, fragment: undefined }
      if (anchor === 'passage') anchor = quote.section ? `#${quote.section.id}` : 'page'
      schedule()
    },
    { signal },
  )
  $<HTMLTextAreaElement>('#press-kit-note')?.addEventListener(
    'input',
    () => {
      noteEdited = true
      schedule()
    },
    { signal },
  )
  modal.addEventListener(
    'click',
    event => {
      const target = event.target instanceof Element ? event.target : null
      const action = target?.closest<HTMLElement>('[data-action]')?.dataset.action
      const copyKind = target?.closest<HTMLElement>('[data-copy]')
      if (action === 'download') void download()
      else if (action === 'copy-image') void copyImage()
      else if (action === 'share-image') void shareImage()
      else if (action === 'share-link') {
        const text = $<HTMLTextAreaElement>('#press-kit-note')?.value.trim()
        void navigator
          .share({ title: titleText(), text: text || undefined, url: shareUrl() })
          .catch(() => {})
      } else if (copyKind) {
        void copy(copyKind.dataset.copy!, copyKind.textContent!.trim())
      }
    },
    { signal },
  )
}

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (activeSignal === signal) return
  activeSignal = signal

  dialog = document.querySelector<HTMLDialogElement>('dialog.press-kit')
  const data = dialog?.querySelector('.press-kit-data')?.textContent
  page = data ? (JSON.parse(data) as PressKitPage) : null
  subject = null
  anchorSelect = null
  unmountAnchor = null
  lastRange = null
  pendingG = -Infinity
  hidePill()
  signal.addEventListener('abort', hidePill, { once: true })
  if (!dialog || !page) return

  bindDialog(signal)
  document.addEventListener('keydown', onDocumentKeyDown, { signal })
  document.addEventListener(
    'presskitopen',
    event => open({ tab: (event as CustomEventMap['presskitopen']).detail?.tab }),
    { signal },
  )
  document.addEventListener(
    'click',
    event => {
      const trigger =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('[data-press-kit]')
          : null
      if (!trigger || trigger.closest('.press-kit')) return
      event.preventDefault()
      const entryId = trigger.dataset.pressKitEntry
      const entry = entryId ? (document.getElementById(entryId) ?? undefined) : undefined
      open({
        tab: (trigger.dataset.pressKit || undefined) as PressKitTab | undefined,
        entry,
        entryUrl: trigger.dataset.pressKitUrl,
      })
    },
    { signal },
  )

  let settle = 0
  const settleSoon = (delay: number) => {
    window.clearTimeout(settle)
    settle = window.setTimeout(onSelectionSettled, delay)
  }
  document.addEventListener(
    'mouseup',
    event => {
      // Modifier selections belong to the comment composer.
      if (event.button !== 0 || event.metaKey || event.altKey || event.ctrlKey) return
      if (event.target instanceof Element && event.target.closest('.press-kit, .press-kit-pill'))
        return
      settleSoon(10)
    },
    { signal },
  )
  document.addEventListener(
    'selectionchange',
    () => {
      const selection = window.getSelection()
      if (!selection || selection.isCollapsed) hidePill()
      // Touch selections have no mouseup; wait for the handles to settle.
      else if (matchMedia('(pointer: coarse)').matches) settleSoon(450)
    },
    { signal },
  )
  signal.addEventListener('abort', () => window.clearTimeout(settle), { once: true })
})
