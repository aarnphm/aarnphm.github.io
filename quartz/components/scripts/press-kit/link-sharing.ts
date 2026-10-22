import { metricShown } from '../press-kit-card'
import { runsToPlain, runsToText } from '../press-kit-text'
import { $, session, toast } from './session-state'
import {
  activeDayBlocks,
  headingText,
  model,
  sectionForCard,
  shareUrl,
  titleText,
} from './share-subject'

const X_LIMIT = 280
// X shortens every link to a t.co URL of this length.
const X_LINK_LENGTH = 23

const display = (url: string) => url.replace(/^https?:\/\//, '').replace(/\/$/, '')

export function defaultNote(): string {
  if (!session.subject) return ''
  if (session.subject.day) {
    const blocks = activeDayBlocks().filter(block =>
      session.anchor.startsWith('block:') ? `block:${block.id}` === session.anchor : true,
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
  const passage =
    session.anchor === 'passage' || session.tab === 'quote' ? runsToPlain(session.quote.runs) : ''
  if (passage && session.quote.chosen) {
    const clipped = passage.length > 180 ? `${passage.slice(0, 177).trimEnd()}…` : passage
    return `“${clipped}” — ${title}`
  }
  const description = runsToPlain(session.subject.description)
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
  const year = (session.subject?.iso ?? new Date().toISOString()).slice(0, 4)
  const family = session.page!.author.split(' ').at(-1)!.toLowerCase()
  const word =
    titleText()
      .toLowerCase()
      .match(/[a-z0-9]+/)?.[0] ?? 'note'
  const given = session.page!.author.split(' ').slice(0, -1).join(' ')
  return [
    `@misc{${family}${year}${word},`,
    `  author = {${session.page!.author.split(' ').at(-1)}, ${given}},`,
    `  title = {{${titleText().replace(/[{}]/g, '')}}},`,
    `  year = {${year}},`,
    `  howpublished = {\\url{${url}}},`,
    `  urldate = {${new Date().toISOString().slice(0, 10)}},`,
    `  note = {${session.page!.site}}`,
    `}`,
  ].join('\n')
}

function copyText(kind: string): string {
  const url = shareUrl()
  const title = titleText()
  const passage = session.anchor === 'passage' ? runsToText(session.quote.runs) : ''
  switch (kind) {
    case 'markdown':
      return passage
        ? `> ${passage.replace(/\n\n/g, '\n>\n> ')}\n>\n> — [${title}](${url})`
        : `[${title}](${url})`
    case 'wikilink': {
      const section = sectionForCard()
      const target =
        session.subject?.kind === 'entry' ? display(url) : session.page!.slug || 'index'
      const hash = session.anchor.startsWith('#') && section ? `#${headingText(section)}` : ''
      return `[[${target}${hash}|${title}]]`
    }
    case 'bibtex':
      return bibtex(url)
    case 'citation': {
      const family = session.page!.author.split(' ').at(-1)
      const given = session.page!.author.split(' ').slice(0, -1).join(' ')
      const date = session.subject?.date ? ` ${session.subject.date}.` : ''
      return `${family}, ${given}. “${title}.” ${session.page!.site},${date} ${url}.`
    }
    default:
      return url
  }
}

export async function copy(kind: string, label: string) {
  try {
    await navigator.clipboard.writeText(copyText(kind))
    toast(`${label} copié`)
  } catch {
    toast('copie impossible')
  }
}

export function renderLinks() {
  const url = shareUrl()
  const input = $<HTMLInputElement>('#press-kit-url')
  if (input) input.value = url
  const note = $<HTMLTextAreaElement>('#press-kit-note')
  if (note && !session.noteEdited) note.value = defaultNote()
  const text = note?.value.trim() ?? ''
  session.dialog?.querySelectorAll<HTMLAnchorElement>('[data-intent]').forEach(link => {
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
    if (session.tab !== 'links' || new URL(shareUrl()).pathname !== path) return
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

export const linkCards = new Map<string, Promise<LinkCard>>()

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
