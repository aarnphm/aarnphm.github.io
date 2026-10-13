import type { Locale } from '../../util/triathlon-presentation'
import type { PressKitPage } from '../PressKit'
import {
  ACTIVITY_CARD_STATS,
  isActivityCardStatKey,
  isActivityCardChartKey,
  OVERLAY_OPTIONS,
  type ActivityCardData,
  type OverlayKey,
  type PowerWindow,
} from '../../util/press-kit-activity'
import { triLabel, triText } from '../../util/triathlon-i18n'
import { buildSelect, type Select } from '../controls/select'
import { PressKitEditor } from '../PressKitEditor'
import { TRI_LOCALE_KEY } from '../triathlon/runtime/preferences'
import { currentNavSignal } from './nav-lifecycle'
import {
  FORMATS,
  cardFaces,
  cardResolution,
  palettes,
  prepareCard,
  prepareCardScene,
  renderCard,
  metricShown,
  type CardModel,
  type FormatKey,
  type Fonts,
  type SiteColors,
  type ThemeKey,
} from './press-kit-card'
import {
  dayCardData,
  blockDisplayKeys,
  isBlockDisplayKey,
  type BlockDisplay,
  type DayCardData,
  type TrainingCardBlock,
} from './press-kit-day'
import { cardLocale, localizeActivity, shortMetricLabel } from './press-kit-i18n'
import { activityCardFromElement } from './press-kit-metrics'
import {
  cardDate,
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
  kind: 'page' | 'entry' | 'activity' | 'day'
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
  activity?: ActivityCardData
  day?: DayCardData
  entryTitle?: string
  entryDate?: string
}

interface Quote {
  runs: Run[]
  attribution?: string
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
  activity?: ActivityCardData
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
let overlay: OverlayKey = 'card'
let language: Locale | null = null
let powerWindow: PowerWindow = 30
let powerCurveScale: 'linear' | 'log' = 'log'
const selectedMetrics = new Map<string, Set<string>>()
const show = {
  art: false,
  description: true,
  excerpt: false,
  meta: true,
  highlight: false,
  section: true,
  title: true,
  date: true,
  metrics: true,
  map: true,
  distance: true,
  time: true,
  pace: true,
  elevation: true,
  heartRate: true,
  power: true,
  entry: false,
  transparent: false,
  cyclingPower: false,
  powerCurve: false,
  cyclingLaps: false,
  runPace: false,
  runLaps: false,
}
let noteEdited = false
let anchorSelect: Select<string> | null = null
let unmountAnchor: (() => void) | null = null
let overlaySelect: Select<OverlayKey> | null = null
let fonts: Fonts | null = null
// The base card by daylight, and its night twin for the encre ground.
const arts: { day?: HTMLImageElement; night?: HTMLImageElement } = {}
let frame = 0
let blob: Promise<Blob | null> | null = null
let blobTimer = 0
let rendering: Promise<void> | null = null
let drawVersion = 0
let renderFailed = false
let editor: PressKitEditor | null = null
const selectedBlocks = new Set<string>()
const selectedActivities = new Set<string>()
const blockDisplays = new Map<string, BlockDisplay>()
let pendingG = -Infinity
let lastRange: { range: Range; at: number } | null = null
// A selection survives this long after it collapses, so a button click elsewhere can still quote it.
const RANGE_TTL_MS = 60_000
let pill: HTMLButtonElement | null = null

const $ = <T extends Element = HTMLElement>(selector: string) =>
  dialog?.querySelector<T>(selector) ?? null

const shareLocale = (): Locale => language ?? cardLocale()

function toast(message: string) {
  document.dispatchEvent(new CustomEvent('toast', { detail: { message } }))
}

function readPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFS_KEY) ?? 'null') as {
      format?: FormatKey
      art?: boolean
    } | null
    if (saved?.format && FORMAT_KEYS.includes(saved.format)) format = saved.format
    if (typeof saved?.art === 'boolean') show.art = saved.art
  } catch {}
  theme = document.documentElement.getAttribute('saved-theme') === 'dark' ? 'encre' : 'papier'
}

function writePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ format, art: show.art }))
  } catch {}
}

function resolveFonts(): Fonts {
  const root = getComputedStyle(document.documentElement)
  const variable = (name: string, fallback: string) =>
    root.getPropertyValue(name).trim() || fallback
  const heading = document.querySelector('h1.article-title')
  const weight = heading ? Number.parseInt(getComputedStyle(heading).fontWeight, 10) : 700
  const header = variable('--headerFont', 'system-ui, sans-serif')
  const headerFamily = header
    .split(',')[0]
    .trim()
    .replace(/^["']|["']$/g, '')
  const availableWeights = Array.from(document.fonts)
    .filter(
      face => face.family.replace(/^["']|["']$/g, '') === headerFamily && face.style === 'normal',
    )
    .map(face => Number(face.weight))
    .filter(Number.isFinite)
  // The page disables synthetic font weights; detached text canvases need the
  // recorded face's weight to render the same heading instead of synthesizing bold.
  const headerWeight = availableWeights.reduce(
    (closest, candidate) =>
      Math.abs(candidate - weight) < Math.abs(closest - weight) ? candidate : closest,
    availableWeights[0] ?? weight,
  )
  return {
    header,
    headerWeight: Number.isFinite(headerWeight) ? headerWeight : 700,
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
    if (fonts === target) {
      layoutMetricControls()
      schedule()
    }
  })
}

// The base social card; same-origin, so drawing it leaves the canvas exportable. The night card
// (quartz/scripts/brand-art.ts) is painted on the encre ground and only ever drawn into cards.
const ART_SRC = '/static/og-image.webp'
const ART_NIGHT_SRC = '/static/og-image-dark.webp'

function loadArt() {
  for (const [key, src] of [
    ['day', ART_SRC],
    ['night', ART_NIGHT_SRC],
  ] as const) {
    if (arts[key]) continue
    const img = new Image()
    img.decoding = 'async'
    img.src = src
    img.onload = () => {
      arts[key] = img
      if (show.art) schedule()
    }
  }
}

function absolute(href: string): string {
  try {
    const url = new URL(href, page!.url)
    // A local preview still shares the canonical address.
    if (url.origin === location.origin)
      return `https://${page!.site}${url.pathname}${url.search}${url.hash}`
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

// reading-time's default rate, which the build uses for a page's minutes.
const readingTime = (text: string): string | undefined => {
  const words = text.split(/\s+/).filter(Boolean).length
  return words > 0 ? `${Math.ceil(words / 200)} min` : undefined
}

// A subject's breadcrumb is its address, as a page lists its slug; a query or hash is not a segment.
const crumbsOf = (url: string): string[] => [
  page!.site,
  ...new URL(url).pathname
    .split('/')
    .filter(Boolean)
    .map(part => part.toLowerCase()),
]

function entrySubject(entry: HTMLElement, href: string | undefined): Subject {
  const title = entry.querySelector('.stream-entry-title')
  const body = entry.querySelector('.stream-entry-body')
  const content = document.createDocumentFragment()
  const collected = dayCardData(entry)
  // A day card's entry block keeps the description in its text; a plain entry sets it apart, as a
  // note sets its description under the title.
  const descriptionNode = collected
    ? null
    : body?.querySelector(':scope > .stream-entry-description')
  body
    ?.querySelectorAll(
      collected
        ? ':scope > .stream-entry-description, :scope > .stream-entry-content'
        : ':scope > .stream-entry-content',
    )
    .forEach(node => content.append(node.cloneNode(true)))
  const day = collected
    ? {
        blocks: collected.blocks.map(block => ({
          ...block,
          url: absolute(block.url),
          activities: block.activities.map(activity => ({
            ...activity,
            url: absolute(activity.url),
          })),
        })),
      }
    : undefined
  if (day)
    content
      .querySelectorAll(
        '.tri-day-embed, .tri-compare-embed, .tri-calc-embed, figure.audio-player, .transclude-src',
      )
      .forEach(node => node.remove())
  const time = entry.querySelector<HTMLTimeElement>(
    '.stream-entry-date time, time.stream-entry-date',
  )
  const url = absolute(href ?? `${location.pathname}#${entry.id}`)
  const excerpt = runsFromNode(content)
  const description = descriptionNode ? runsFromNode(descriptionNode) : []
  const activities =
    day?.blocks.filter(block => block.kind === 'activity').flatMap(block => block.activities) ?? []
  return {
    kind: day ? 'day' : 'entry',
    entryTitle: title?.textContent?.trim(),
    entryDate: time?.dateTime ? cardDate(time.dateTime, shareLocale()) : undefined,
    url,
    title: activities.length
      ? runsFromText(activities.map(activity => activity.title).join(' + '))
      : title
        ? runsFromNode(title)
        : [],
    description,
    excerpt,
    date: (() => {
      // A training day carries the day it trained, which can precede the entry's own day.
      const iso = day
        ? (entry.querySelector<HTMLElement>('.tri-day-embed')?.dataset.triathlonDate ??
          day.blocks.flatMap(block => block.activities)[0]?.date ??
          time?.dateTime)
        : time?.dateTime
      return iso ? cardDate(iso, shareLocale()) : time?.textContent?.trim() || undefined
    })(),
    iso: time?.dateTime || undefined,
    reading: day ? undefined : readingTime(`${runsToPlain(description)} ${runsToPlain(excerpt)}`),
    tags: Array.from(entry.querySelectorAll('.stream-entry-tag'), tag => tag.textContent!.trim())
      .filter(Boolean)
      .slice(0, 4),
    crumbs: crumbsOf(url),
    headings: [],
    day,
  }
}

function activitySubject(activity: ActivityCardData): Subject {
  const url = absolute(activity.url)
  return {
    kind: 'activity',
    url,
    title: runsFromText(activity.title),
    description: [],
    excerpt: runsFromText(activity.stats.map(stat => `${stat.label}: ${stat.value}`).join('\n')),
    date: cardDate(activity.date, shareLocale()),
    iso: activity.date,
    tags: [activity.sport],
    crumbs: crumbsOf(url),
    headings: [],
    activity: { ...activity, url: absolute(activity.url) },
  }
}

function citationAllowed(): boolean {
  const slug = page?.slug ?? ''
  return (
    subject?.kind !== 'activity' &&
    subject?.kind !== 'day' &&
    slug !== 'stream' &&
    !slug.startsWith('stream/on') &&
    slug !== 'triathlon' &&
    !slug.startsWith('triathlon/')
  )
}

function availableTabs(): PressKitTab[] {
  return citationAllowed() ? TABS : ['note', 'links']
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
  if (anchor.startsWith('block:')) {
    const block = subject.day?.blocks.find(block => `block:${block.id}` === anchor)
    if (block) return absolute(block.url)
  }
  const base = subject.url.split('#')[0]
  if (anchor === 'passage' && quote.fragment) {
    const section = quote.section ? `#${quote.section.id}` : '#'
    return `${base}${section}:~:${quote.fragment}`
  }
  if (anchor.startsWith('#')) return `${base}${anchor}`
  return subject.url
}

function model(): CardModel {
  const s = subject!
  const locale = shareLocale()
  const section = sectionForCard()
  const kind = tab === 'quote' ? 'quote' : s.kind === 'page' ? 'note' : s.kind
  const attribution = kind === 'quote' ? quote.attribution?.trim() : undefined
  const day = s.day
    ? {
        blocks: activeDayBlocks().map(block => ({
          ...block,
          title:
            block.kind === 'memo'
              ? locale === 'fr'
                ? 'mémo audio'
                : 'audio memo'
              : block.kind === 'comparison'
                ? `${triText(locale, 'comparison')} · ${[...new Set(block.activities.map(activity => activity.date))].join(' / ')}`
                : block.title,
          rows: block.rows.map(row => ({ ...row, label: triLabel(locale, row.label) })),
          activities: block.activities.map(activity => localizeActivity(activity, locale)),
          display: blockDisplays.get(block.id),
        })),
      }
    : undefined
  const entry = runsToPlain(s.excerpt).trim()
  if (day && show.entry && entry)
    day.blocks.unshift({
      id: 'entry',
      kind: 'entry',
      title: s.entryTitle ?? 'entrée',
      url: s.url,
      activities: [],
      rows: [],
      text: entry,
      date: s.entryDate,
      reading: readingTime(entry),
      display: blockDisplays.get('entry'),
    })
  return {
    kind,
    locale,
    crumbs: s.crumbs,
    title: attribution
      ? [{ text: attribution, style: 'roman' }]
      : s.title.length > 0
        ? s.title
        : [{ text: page!.site, style: 'roman' }],
    description: s.description,
    body: tab === 'quote' ? quote.runs : s.excerpt,
    date: s.date,
    reading: s.reading,
    tags: s.tags,
    section: section ? headingText(section) : undefined,
    art: theme === 'encre' ? arts.night : arts.day,
    // The night card's moon sits over the camp, so its crops centre higher.
    artFocus: theme === 'encre' ? [640, 280] : undefined,
    activity: s.activity ? localizeActivity(s.activity, locale) : undefined,
    day,
    overlay,
    metricGap: editor?.metricGap ?? 0,
    powerWindow,
    powerCurveScale,
    metricKeys: Object.fromEntries(Array.from(selectedMetrics, ([url, keys]) => [url, [...keys]])),
    show: {
      ...show,
      meta: kind === 'activity' || kind === 'day' || show.meta,
      excerpt: kind === 'entry' || show.excerpt,
    },
  }
}

function caption() {
  const spec = FORMATS[format]
  const resolution = cardResolution(format)
  const el = $('.press-kit-caption')
  if (!el) return
  el.textContent =
    tab === 'links'
      ? 'aperçu du lien partagé'
      : `${resolution.width} × ${resolution.height} · png · ${spec.label}`
}

function draw() {
  frame = 0
  const version = ++drawVersion
  rendering = null
  window.clearTimeout(blobTimer)
  blob = null
  renderFailed = false
  if (!dialog?.open || !subject || !fonts || !page) return
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  const unfurl = $('.press-kit-unfurl')
  if (!canvas || !unfurl) return
  dialog.dataset.format = format
  caption()
  editor?.context(`${subject.url}:${tab}`, `${format}:${overlay}`, tab !== 'links')
  if (tab === 'links') {
    canvas.hidden = true
    unfurl.hidden = false
    renderLinks()
    return
  }
  canvas.hidden = false
  unfurl.hidden = true
  const colors = page.colors as unknown as { light: SiteColors; dark: SiteColors }
  const palette = palettes(colors.light, colors.dark)[theme]
  const targetFormat = format
  const targetFonts = fonts
  const targetDialog = dialog
  canvas.setAttribute('aria-label', `aperçu de la carte, ${FORMATS[format].label}, ${theme}`)
  canvas.setAttribute('aria-busy', 'true')
  const targetEditor = editor
  rendering = prepareCard(model(), palette, targetFormat, targetFonts, targetEditor?.active)
    .then(async prepared => {
      if (version !== drawVersion || dialog !== targetDialog || !dialog.open) return
      if (targetEditor?.active) {
        const scene = await prepareCardScene(
          prepared,
          targetFormat,
          palette,
          targetFonts,
          activeSignal ?? undefined,
        )
        if (version !== drawVersion || dialog !== targetDialog || !dialog.open) return
        targetEditor.setScene(scene)
      } else renderCard(canvas, prepared, targetFormat, palette, targetFonts)
      blobTimer = window.setTimeout(() => {
        blob = new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
      }, 160)
    })
    .catch(() => {
      if (version !== drawVersion) return
      renderFailed = true
      toast('impossible de rendre la carte, veuillez réessayer')
    })
    .finally(() => {
      if (version !== drawVersion) return
      canvas.setAttribute('aria-busy', 'false')
      rendering = null
    })
}

function schedule() {
  if (!frame) frame = requestAnimationFrame(draw)
}

async function currentBlob(): Promise<Blob | null> {
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  const targetDialog = dialog
  const targetEditor = editor
  do {
    if (frame) {
      cancelAnimationFrame(frame)
      draw()
    }
    if (rendering) await rendering
    if ((await targetEditor?.flush()) === false) return null
    if (!canvas?.isConnected || dialog !== targetDialog || !dialog?.open || editor !== targetEditor)
      return null
  } while (frame || rendering)
  if (renderFailed) return null
  if (!blob) {
    window.clearTimeout(blobTimer)
    blob = new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  }
  return blob
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

const titleText = () =>
  subject?.day?.blocks.find(block => `block:${block.id}` === anchor)?.title ??
  (runsToPlain(subject?.title ?? []) || page?.site || '')

function defaultNote(): string {
  if (!subject) return ''
  if (subject.day) {
    const blocks = activeDayBlocks().filter(block =>
      anchor.startsWith('block:') ? `block:${block.id}` === anchor : true,
    )
    return [
      titleText(),
      ...blocks.map(block =>
        block.kind === 'activity'
          ? `${block.title} · ${block.activities[0].stats
              .filter(stat => metricShown(model(), block.activities[0], stat))
              .map(stat => stat.value)
              .join(' · ')}`
          : block.title,
      ),
    ].join('\n')
  }
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
  const options: { value: string; label: string }[] = [
    {
      value: 'page',
      label:
        subject?.kind === 'activity'
          ? 'activité entière'
          : subject?.kind === 'day'
            ? 'journée entière'
            : subject?.kind === 'entry'
              ? 'entrée entière'
              : 'page entière',
    },
  ]
  if (quote.fragment) options.push({ value: 'passage', label: 'passage sélectionné' })
  const headings = (subject?.headings ?? []).map(heading => ({
    value: `#${heading.id}`,
    label: `${'  '.repeat(Math.max(0, Number(heading.tagName[1]) - 2))}${headingText(heading)}`,
  }))
  const blocks =
    subject?.day?.blocks.map(block => ({ value: `block:${block.id}`, label: block.title })) ?? []
  return [
    { options },
    ...(blocks.length ? [{ label: 'blocs', options: blocks }] : []),
    ...(headings.length > 0 ? [{ label: 'sections', options: headings }] : []),
  ]
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

function mountOverlay(signal: AbortSignal) {
  const host = $('.press-kit-overlay')
  if (!host) return
  overlaySelect = buildSelect<OverlayKey>({
    id: 'press-kit-overlay-menu',
    label: 'superposition',
    groups: [{ options: OVERLAY_OPTIONS.map(([value, label]) => ({ value, label })) }],
    selected: overlay,
    className: 'press-kit-select',
    onSelect: value => {
      overlay = value
      schedule()
    },
  })
  host.replaceChildren(overlaySelect.element)
  signal.addEventListener('abort', overlaySelect.mount(), { once: true })
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
  const site = $('.press-kit-unfurl-site')
  if (site) site.textContent = display(url).split('/')[0]
  const path = new URL(url).pathname
  void linkCard(path).then(card => {
    if (tab !== 'links' || new URL(shareUrl()).pathname !== path) return
    const image = $<HTMLImageElement>('.press-kit-unfurl-image')
    if (image && image.getAttribute('src') !== card.image) {
      // A missing card image collapses the preview to its text rather than a broken frame.
      image.hidden = !card.image
      image.onerror = () => (image.hidden = true)
      if (card.image) image.src = card.image
      else image.removeAttribute('src')
    }
    const title = $('.press-kit-unfurl-title')
    if (title) title.textContent = card.title || titleText()
  })
}

interface LinkCard {
  image?: string
  title?: string
}

const linkCards = new Map<string, Promise<LinkCard>>()

const linkCardOf = (head: ParentNode): LinkCard => ({
  image: head.querySelector<HTMLMetaElement>('meta[property="og:image"]')?.content,
  title: head.querySelector<HTMLMetaElement>('meta[property="og:title"]')?.content,
})

/**
 * The card a social site unfurls is the target page's own og:image and og:title, so the preview
 * reads them from that page. A training day page runs to megabytes; the read stops at `</head>`.
 */
function linkCard(path: string): Promise<LinkCard> {
  if (path === location.pathname) return Promise.resolve(linkCardOf(document.head))
  let card = linkCards.get(path)
  if (!card) {
    card = (async () => {
      const response = await fetch(path, { headers: { Accept: 'text/html' } })
      if (!response.ok || !response.body) return {}
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let html = ''
      while (!html.includes('</head>')) {
        const { done, value } = await reader.read()
        if (done) break
        html += decoder.decode(value, { stream: true })
      }
      void reader.cancel()
      return linkCardOf(new DOMParser().parseFromString(html, 'text/html'))
    })().catch(() => ({}))
    linkCards.set(path, card)
  }
  return card
}

const HELP: Record<'image' | 'links', string> = {
  image: '{1}{2}{3} onglets · {f} format · {t} thème · {d} télécharger · {c} copier · {esc} fermer',
  links: '{1}{2}{3} onglets · {esc} fermer',
}

function setHelp() {
  const status = $('.press-kit-help')
  if (!status) return
  const template = HELP[tab === 'links' ? 'links' : 'image'].replace(
    '{3}',
    citationAllowed() ? '{3}' : '',
  )
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
  dialog.dataset.training = String(subject?.kind === 'activity' || subject?.kind === 'day')
  dialog.dataset.transparent = String(show.transparent)
  dialog.dataset.citation = String(citationAllowed())
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
  overlaySelect?.setSelected(overlay)
  dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-language"]')
    .forEach(input => (input.checked = input.value === shareLocale()))
  const activities = activeActivities()
  const blocks = activeDayBlocks()
  dialog.querySelectorAll<HTMLButtonElement>('[data-include]').forEach(button => {
    if (isIncludeKey(button.name)) {
      const key = button.name
      const values = isBlockDisplayKey(key)
        ? Array.from(blockDisplays.entries())
            .filter(([id]) =>
              id === 'entry'
                ? show.entry
                : blocks.some(block => block.id === id && blockDisplayKeys(block).includes(key)),
            )
            .map(([, display]) => display[key])
        : []
      button.setAttribute(
        'aria-pressed',
        values.length && values.some(Boolean) && values.some(value => !value)
          ? 'mixed'
          : String(values.length ? values.every(Boolean) : show[key]),
      )
    }
    if (button.closest('[data-activity-only]')) {
      button.disabled =
        button.name === 'map'
          ? !activities.some(activity => activity.route.length > 0)
          : ACTIVITY_CARD_STATS.some(key => key === button.name) &&
            !activities.some(activity => activity.stats.some(stat => stat.key === button.name))
    }
    if (isActivityCardChartKey(button.name)) {
      const available = activities.some(activity =>
        activity.charts.some(chart => chart.key === button.name),
      )
      button.disabled = !available
      button.hidden = !available
    }
    if (button.name === 'entry') button.disabled = !runsToPlain(subject?.excerpt ?? []).trim()
  })
  dialog.querySelectorAll<HTMLButtonElement>('[data-block-display]').forEach(button => {
    const key = button.dataset.blockDisplay
    const id = button.dataset.displayBlock
    if (!key || !id || !isBlockDisplayKey(key)) return
    const display = blockDisplays.get(id)
    button.setAttribute('aria-pressed', String(display?.[key] ?? show[key]))
    button.disabled = id === 'entry' ? !show.entry : !blocks.some(block => block.id === id)
  })
  const charts = $<HTMLButtonElement>('[data-include="cyclingPower"]')?.closest('fieldset')
  if (charts) charts.hidden = !activities.some(activity => activity.charts.length > 0)
  const powerCharts = activities.flatMap(activity =>
    activity.charts.filter(chart => chart.key === 'cyclingPower'),
  )
  const availableWindow = (seconds: number) =>
    powerCharts.length > 0 &&
    powerCharts.every(
      chart => chart.series?.some(series => series.seconds === seconds) ?? seconds === 30,
    )
  if (!availableWindow(powerWindow)) powerWindow = 30
  const powerControls = $('.press-kit-power-window')
  if (powerControls) powerControls.hidden = !show.cyclingPower || !powerCharts.length
  const powerScale = $('.press-kit-power-scale')
  if (powerScale)
    powerScale.hidden =
      !show.powerCurve ||
      !activities.some(activity => activity.charts.some(chart => chart.key === 'powerCurve'))
  dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-power-scale"]')
    .forEach(input => {
      input.checked = input.value === powerCurveScale
    })
  dialog
    .querySelectorAll<HTMLInputElement>('input[name="press-kit-power-window"]')
    .forEach(input => {
      input.checked = Number(input.value) === powerWindow
      input.disabled = !availableWindow(Number(input.value))
    })
  dialog.querySelectorAll<HTMLButtonElement>('[data-day-activity]').forEach(button => {
    const selected = selectedActivities.has(button.dataset.dayActivity ?? '')
    button.setAttribute('aria-pressed', String(selected))
    button.title = selected ? 'masquer cette activité' : 'inclure cette activité'
  })
  dialog.querySelectorAll<HTMLButtonElement>('[data-day-block]').forEach(button => {
    const block = subject?.day?.blocks.find(block => block.id === button.dataset.dayBlock)
    button.disabled =
      block?.kind === 'comparison' &&
      block.activities.filter(activity => selectedActivities.has(activity.url)).length < 2
  })
  const entryDisplay = $('.press-kit-entry-display')
  if (entryDisplay) entryDisplay.hidden = !show.entry
  dialog.querySelectorAll<HTMLButtonElement>('[data-activity-metric]').forEach(button => {
    const url = button.dataset.metricActivity
    const key = button.dataset.activityMetric
    button.setAttribute(
      'aria-pressed',
      String(Boolean(url && key && selectedMetrics.get(url)?.has(key))),
    )
  })
  dialog
    .querySelectorAll<HTMLElement>('[data-metric-group]')
    .forEach(
      group =>
        (group.hidden = !activities.some(activity => activity.url === group.dataset.metricGroup)),
    )
  const textarea = $<HTMLTextAreaElement>('#press-kit-quote')
  if (textarea && document.activeElement !== textarea) textarea.value = runsToText(quote.runs)
  const attribution = $<HTMLInputElement>('#press-kit-attribution')
  if (attribution) {
    attribution.placeholder = titleText()
    if (document.activeElement !== attribution) attribution.value = quote.attribution ?? ''
  }
  anchorSelect?.setSelected(anchor)
  setHelp()
}

function subjectActivities(): ActivityCardData[] {
  const activities =
    subject?.day?.blocks.flatMap(block => block.activities) ??
    (subject?.activity ? [subject.activity] : [])
  const unique = new Map<string, ActivityCardData>()
  for (const activity of activities) {
    const previous = unique.get(activity.url)
    if (!previous || previous.stats.length < activity.stats.length)
      unique.set(activity.url, activity)
  }
  return [...unique.values()]
}

function activeDayBlocks(): TrainingCardBlock[] {
  return (subject?.day?.blocks ?? [])
    .filter(block => selectedBlocks.has(block.id))
    .flatMap(block => {
      if (!block.activities.length) return [block]
      const activities = block.activities.filter(activity => selectedActivities.has(activity.url))
      if (!activities.length || (block.kind === 'comparison' && activities.length < 2)) return []
      if (activities.length === block.activities.length) return [block]
      return [{ ...block, activities, graphic: undefined, rows: [] }]
    })
}

function activeActivities(): ActivityCardData[] {
  return subject?.day
    ? activeDayBlocks().flatMap(block => block.activities)
    : subject?.activity
      ? [subject.activity]
      : []
}

function syncMetricIncludes() {
  for (const key of ACTIVITY_CARD_STATS)
    show[key] = subjectActivities().some(activity => selectedMetrics.get(activity.url)?.has(key))
}

function layoutMetricControls() {
  if (!dialog?.open) return
  const context = document.createElement('canvas').getContext('2d')
  if (!context) return
  const textWidth = (element: HTMLElement): number => {
    const style = getComputedStyle(element)
    context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
    return context.measureText(element.textContent ?? '').width
  }
  for (const metrics of dialog.querySelectorAll<HTMLElement>(
    '.press-kit-metric-group[open] .press-kit-data-grid',
  )) {
    if (!metrics.clientWidth) continue
    metrics.querySelector('.press-kit-data-filler')?.remove()
    const columnWidth = (metrics.clientWidth - parseFloat(getComputedStyle(metrics).columnGap)) / 2
    const items = Array.from(metrics.querySelectorAll<HTMLButtonElement>('[data-activity-metric]'))
      .map(button => {
        const label = button.querySelector<HTMLElement>('.press-kit-metric-label')
        const value = button.querySelector<HTMLElement>('small')
        if (!label || !value) return null
        const style = getComputedStyle(button)
        const inset =
          parseFloat(style.paddingInlineStart) +
          parseFloat(style.paddingInlineEnd) +
          parseFloat(style.borderInlineStartWidth) +
          parseFloat(style.borderInlineEndWidth)
        const available = columnWidth - inset
        const wide = textWidth(label) > available || textWidth(value) > available
        button.classList.toggle('press-kit-data-toggle--wide', wide)
        return { button, wide, index: Number(button.dataset.metricOrder ?? 0) }
      })
      .filter(item => item !== null)
      .sort((a, b) => Number(a.wide) - Number(b.wide) || a.index - b.index)
    if (items.some((item, index) => metrics.children[index] !== item.button)) {
      const focus = document.activeElement
      metrics.append(...items.map(item => item.button))
      if (focus instanceof HTMLElement && metrics.contains(focus))
        focus.focus({ preventScroll: true })
    }
    // An odd count of half-width toggles leaves a hole before the full-width rows; an empty
    // cell closes the table there.
    const narrow = items.filter(item => !item.wide)
    if (narrow.length % 2) {
      const filler = document.createElement('span')
      filler.className = 'press-kit-data-filler'
      filler.setAttribute('aria-hidden', 'true')
      narrow[narrow.length - 1].button.after(filler)
    }
  }
}

function refreshLocale() {
  if (!subject) return
  const locale = shareLocale()
  if (subject.activity) subject.activity = localizeActivity(subject.activity, locale)
  if (subject.day)
    subject.day.blocks = subject.day.blocks.map(block => ({
      ...block,
      rows: block.rows.map(row => ({ ...row, label: triLabel(locale, row.label) })),
      activities: block.activities.map(activity => localizeActivity(activity, locale)),
    }))
  if (subject.iso) subject.entryDate = cardDate(subject.iso, locale)
  if (subject.kind === 'entry' && subject.iso) subject.date = cardDate(subject.iso, locale)
  if (subject.kind === 'activity' && subject.activity)
    subject.date = cardDate(subject.activity.date, locale)
  if (subject.kind === 'day') {
    const date = subject.day?.blocks.flatMap(block => block.activities)[0]?.date ?? subject.iso
    if (date) subject.date = cardDate(date, locale)
  }
  mountMetricControls()
  syncControls()
  layoutMetricControls()
  if (dialog?.open) schedule()
}

function mountMetricControls() {
  const host = $('.press-kit-metrics')
  if (!host) return
  const expanded = new Set(
    Array.from(
      host.querySelectorAll<HTMLElement>('[data-metric-group][open]'),
      group => group.dataset.metricGroup,
    ),
  )
  host.replaceChildren()
  for (const source of subjectActivities()) {
    const activity = localizeActivity(source, shareLocale())
    const previous = selectedMetrics.get(activity.url)
    selectedMetrics.set(
      activity.url,
      new Set(
        activity.stats
          .filter(stat =>
            previous ? previous.has(stat.key) : isActivityCardStatKey(stat.key) && show[stat.key],
          )
          .map(stat => stat.key),
      ),
    )
    const group = document.createElement('details')
    group.className = 'press-kit-metric-group'
    group.dataset.metricGroup = activity.url
    group.open = expanded.has(activity.url)
    const summary = document.createElement('summary')
    summary.textContent = `mesures · ${activity.title}`
    group.append(summary)
    const actions = document.createElement('div')
    actions.className = 'press-kit-metric-actions'
    for (const [action, label] of [
      ['all', 'tout'],
      ['none', 'aucune'],
    ]) {
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'g-button'
      button.dataset.metricAction = action
      button.dataset.metricActivity = activity.url
      button.textContent = label
      actions.append(button)
    }
    group.append(actions)
    const metrics = document.createElement('div')
    metrics.className = 'press-kit-data-grid'
    for (const [index, stat] of activity.stats.entries()) {
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'press-kit-data-toggle'
      button.dataset.activityMetric = stat.key
      button.dataset.metricActivity = activity.url
      button.dataset.metricOrder = String(index)
      button.title = [stat.label, stat.value, stat.provenance].filter(Boolean).join(' · ')
      button.setAttribute('aria-label', `${activity.title} · ${stat.label}`)
      button.setAttribute('aria-pressed', String(selectedMetrics.get(activity.url)?.has(stat.key)))
      const label = document.createElement('span')
      label.className = 'press-kit-data-label'
      const labelText = document.createElement('span')
      labelText.className = 'press-kit-metric-label'
      labelText.textContent = shortMetricLabel(stat.label)
      label.append(labelText)
      button.append(label)
      const value = document.createElement('small')
      value.textContent = stat.value
      button.append(value)
      metrics.append(button)
    }
    group.append(metrics)
    host.append(group)
  }
  if (subjectActivities().length) syncMetricIncludes()
}

function displayControls(block: TrainingCardBlock): HTMLDetailsElement {
  const group = document.createElement('details')
  group.className = 'press-kit-block-display'
  group.dataset.blockOptions = block.id
  const summary = document.createElement('summary')
  summary.textContent = `affichage · ${block.kind === 'entry' ? 'entrée' : block.title}`
  group.append(summary)
  const options = document.createElement('div')
  options.className = 'press-kit-data-grid press-kit-data-grid--compact'
  blockDisplays.set(block.id, { title: show.title, date: show.date, metrics: show.metrics })
  for (const key of blockDisplayKeys(block)) {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'press-kit-data-toggle'
    button.dataset.blockDisplay = key
    button.dataset.displayBlock = block.id
    const label = key === 'title' ? 'titre' : key === 'date' ? 'date' : 'mesures'
    button.setAttribute(
      'aria-label',
      `${block.kind === 'entry' ? 'entrée' : block.title} · ${label}`,
    )
    const text = document.createElement('span')
    text.className = 'press-kit-data-label'
    text.textContent = label
    button.append(text)
    options.append(button)
  }
  group.append(options)
  return group
}

function mountDayControls() {
  const host = $('.press-kit-day-blocks')
  if (!host) return
  host.replaceChildren()
  selectedBlocks.clear()
  selectedActivities.clear()
  blockDisplays.clear()
  const entryDisplay = $('.press-kit-entry-display')
  entryDisplay?.replaceChildren()
  if (subject?.day && entryDisplay) {
    const prose = runsToPlain(subject.excerpt).trim()
    entryDisplay.append(
      displayControls({
        id: 'entry',
        kind: 'entry',
        title: subject.entryTitle ?? 'entrée',
        url: subject.url,
        text: prose,
        date: subject.entryDate,
        reading: readingTime(prose),
        activities: [],
        rows: [],
      }),
    )
  }
  const blocks = subject?.day?.blocks ?? []
  const activities = subject?.day ? subjectActivities() : []
  if (activities.length) {
    const field = document.createElement('fieldset')
    field.className = 'press-kit-field'
    const legend = document.createElement('legend')
    legend.textContent = 'activités'
    const list = document.createElement('div')
    list.className = 'press-kit-data-grid press-kit-block-list'
    field.append(legend, list)
    host.append(field)
    for (const activity of activities) {
      selectedActivities.add(activity.url)
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'press-kit-data-toggle'
      button.dataset.dayActivity = activity.url
      button.setAttribute('aria-pressed', 'true')
      const label = document.createElement('span')
      label.className = 'press-kit-data-label'
      label.textContent = activity.title
      button.append(label)
      list.append(button)
      const block = blocks.find(
        block => block.kind === 'activity' && block.activities[0]?.url === activity.url,
      )
      if (block) list.append(displayControls(block))
    }
  }
  for (const block of blocks) {
    const selected = block.kind !== 'memo'
    if (selected) selectedBlocks.add(block.id)
    if (block.kind === 'activity') continue
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'press-kit-data-toggle'
    button.dataset.dayBlock = block.id
    button.setAttribute('aria-pressed', String(selected))
    const label = document.createElement('span')
    label.className = 'press-kit-data-label'
    label.textContent = block.title
    button.append(label)
    host.append(button)
    host.append(displayControls(block))
  }
}

function setTab(next: PressKitTab, focus = false) {
  tab = availableTabs().includes(next) ? next : 'note'
  syncControls()
  layoutMetricControls()
  if (focus) $<HTMLButtonElement>(`#press-kit-tab-${tab}`)?.focus()
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
  loadArt()

  subject = request.activity
    ? activitySubject(request.activity)
    : request.entry
      ? entrySubject(request.entry, request.entryUrl)
      : pageSubject()
  mountDayControls()
  mountMetricControls()
  const title = $('#press-kit-title')
  if (title)
    title.textContent =
      subject.kind === 'activity'
        ? 'kit de presse · activité'
        : subject.kind === 'day'
          ? 'kit de presse · journée'
          : subject.kind === 'entry'
            ? 'kit de presse · entrée'
            : 'kit de presse'
  const recent =
    lastRange && performance.now() - lastRange.at < RANGE_TTL_MS ? lastRange.range : null
  const candidate = request.range ?? selectedRange() ?? recent
  // An entry card only quotes text from inside that entry.
  const usable =
    citationAllowed() &&
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
  if (!availableTabs().includes(tab)) tab = 'note'

  mountAnchor(currentNavSignal())
  const shareImageButton = $<HTMLButtonElement>('[data-action="share-image"]')
  if (shareImageButton) shareImageButton.hidden = !canShareFiles()
  const shareLinkButton = $<HTMLButtonElement>('[data-action="share-link"]')
  if (shareLinkButton) shareLinkButton.hidden = typeof navigator.share !== 'function'

  hidePill()
  syncControls()
  dialog.showModal()
  layoutMetricControls()
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
      setTab(cycle(availableTabs(), tab, key === 'ArrowRight' ? 1 : -1), true)
      event.preventDefault()
      return
    }
    if (key === 'Home' || key === 'End') {
      setTab(key === 'Home' ? availableTabs()[0] : availableTabs().at(-1)!, true)
      event.preventDefault()
      return
    }
  }
  if (isEditable(event.target)) return
  const imageTab = tab !== 'links'
  if (key === '1' || key === '2' || key === '3') {
    const next = availableTabs()[Number(key) - 1]
    if (!next) return
    setTab(next, true)
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
  if (dialog?.open || !citationAllowed()) return
  const range = selectedRange()
  if (!range) {
    hidePill()
    return
  }
  lastRange = { range, at: performance.now() }
  showPill(range)
}

function isIncludeKey(name: string): name is keyof CardModel['show'] {
  return Object.hasOwn(show, name)
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
  const close = $('.press-kit-close')
  if (close) {
    close.dataset.siteCursorClose = ''
    close.addEventListener('click', () => modal.close(), { signal })
  }
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
  modal.querySelectorAll<HTMLButtonElement>('[data-include]').forEach(button => {
    button.addEventListener(
      'click',
      () => {
        if (!isIncludeKey(button.name)) return
        show[button.name] = isBlockDisplayKey(button.name)
          ? button.getAttribute('aria-pressed') !== 'true'
          : !show[button.name]
        const displayKey = button.name
        if (isBlockDisplayKey(displayKey))
          for (const display of blockDisplays.values()) display[displayKey] = show[displayKey]
        if (isActivityCardStatKey(button.name)) {
          for (const keys of selectedMetrics.values()) {
            if (show[button.name]) keys.add(button.name)
            else keys.delete(button.name)
          }
        }
        button.setAttribute('aria-pressed', String(show[button.name]))
        syncControls()
        writePrefs()
        schedule()
      },
      { signal },
    )
  })
  modal.addEventListener(
    'change',
    event => {
      const input = event.target
      if (!(input instanceof HTMLInputElement)) return
      if (input.name === 'press-kit-language' && (input.value === 'en' || input.value === 'fr')) {
        language = input.value
        refreshLocale()
        return
      }
      if (input.name === 'press-kit-format') format = input.value as FormatKey
      else if (
        input.name === 'press-kit-power-window' &&
        (input.value === '30' || input.value === '300')
      ) {
        powerWindow = input.value === '30' ? 30 : 300
        schedule()
        return
      } else if (
        input.name === 'press-kit-power-scale' &&
        (input.value === 'linear' || input.value === 'log')
      ) {
        powerCurveScale = input.value
        schedule()
        return
      } else if (input.name === 'press-kit-theme') theme = input.value as ThemeKey
      else return
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
  $<HTMLInputElement>('#press-kit-attribution')?.addEventListener(
    'input',
    event => {
      if (!(event.target instanceof HTMLInputElement)) return
      quote = { ...quote, attribution: event.target.value }
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
      const displayButton = target?.closest<HTMLButtonElement>('[data-block-display]')
      const displayKey = displayButton?.dataset.blockDisplay
      const displayBlock = displayButton?.dataset.displayBlock
      if (displayButton && displayKey && displayBlock && isBlockDisplayKey(displayKey)) {
        const display = blockDisplays.get(displayBlock)
        if (!display) return
        display[displayKey] = !display[displayKey]
        syncControls()
        schedule()
        return
      }
      const metricButton = target?.closest<HTMLButtonElement>(
        '[data-activity-metric], [data-metric-action]',
      )
      const metricUrl = metricButton?.dataset.metricActivity
      if (metricButton && metricUrl) {
        const activity = subjectActivities().find(activity => activity.url === metricUrl)
        const keys = selectedMetrics.get(metricUrl)
        if (!activity || !keys) return
        const key = metricButton.dataset.activityMetric
        if (key) {
          if (keys.has(key)) keys.delete(key)
          else keys.add(key)
        } else {
          keys.clear()
          if (metricButton.dataset.metricAction === 'all')
            for (const stat of activity.stats) keys.add(stat.key)
        }
        syncMetricIncludes()
        syncControls()
        schedule()
        return
      }
      const activityButton = target?.closest<HTMLButtonElement>('[data-day-activity]')
      const activityUrl = activityButton?.dataset.dayActivity
      if (activityButton && activityUrl) {
        if (selectedActivities.has(activityUrl)) selectedActivities.delete(activityUrl)
        else selectedActivities.add(activityUrl)
        if (
          anchor.startsWith('block:') &&
          !activeDayBlocks().some(block => `block:${block.id}` === anchor)
        )
          anchor = 'page'
        syncControls()
        schedule()
        return
      }
      const blockButton = target?.closest<HTMLButtonElement>('[data-day-block]')
      const blockId = blockButton?.dataset.dayBlock
      if (blockButton && blockId) {
        if (selectedBlocks.has(blockId)) selectedBlocks.delete(blockId)
        else selectedBlocks.add(blockId)
        blockButton.setAttribute('aria-pressed', String(selectedBlocks.has(blockId)))
        syncControls()
        schedule()
        return
      }
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
  editor = null
  drawVersion++
  rendering = null

  dialog = document.querySelector<HTMLDialogElement>('dialog.press-kit')
  const data = dialog?.querySelector('.press-kit-data')?.textContent
  page = data ? (JSON.parse(data) as PressKitPage) : null
  subject = null
  selectedBlocks.clear()
  selectedActivities.clear()
  blockDisplays.clear()
  selectedMetrics.clear()
  linkCards.clear()
  overlay = 'card'
  language = null
  show.transparent = false
  show.entry = false
  show.cyclingPower = false
  show.powerCurve = false
  show.cyclingLaps = false
  powerWindow = 30
  powerCurveScale = 'log'
  show.runPace = false
  show.runLaps = false
  anchorSelect = null
  unmountAnchor = null
  overlaySelect = null
  lastRange = null
  pendingG = -Infinity
  hidePill()
  signal.addEventListener('abort', hidePill, { once: true })
  if (!dialog || !page) return

  bindDialog(signal)
  mountOverlay(signal)
  dialog.addEventListener('toggle', layoutMetricControls, { capture: true, signal })
  const controls = $('.press-kit-controls')
  const canvas = $<HTMLCanvasElement>('.press-kit-canvas')
  if (controls && canvas)
    editor = new PressKitEditor(
      canvas,
      controls,
      signal,
      () => {
        window.clearTimeout(blobTimer)
        blob = null
      },
      schedule,
    )
  if (controls) {
    let width = 0
    const observer = new ResizeObserver(entries => {
      const next = entries[0]?.contentRect.width ?? 0
      if (next === width) return
      width = next
      layoutMetricControls()
    })
    observer.observe(controls)
    window.addCleanup(() => observer.disconnect())
  }
  document.addEventListener('keydown', onDocumentKeyDown, { signal })
  window.addEventListener('tri:locale', refreshLocale, { signal })
  window.addEventListener(
    'storage',
    event => {
      if (event.key === TRI_LOCALE_KEY) refreshLocale()
    },
    { signal },
  )
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
      const entry = entryId
        ? (trigger.closest<HTMLElement>('.stream-entry') ??
          document.getElementById(entryId) ??
          undefined)
        : undefined
      const activityData = trigger.dataset.pressKitActivity
      const activity = activityCardFromElement(trigger)
      if (activityData && !activity) return
      open({
        tab: (trigger.dataset.pressKit || undefined) as PressKitTab | undefined,
        entry,
        entryUrl: trigger.dataset.pressKitUrl,
        activity: activity ?? undefined,
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
