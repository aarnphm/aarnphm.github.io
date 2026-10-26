import { normalizeRelativeURLs } from '../../../util/path'
import { setStatusText } from '../status-flap'

interface Note {
  title: string
  body: DocumentFragment
}

/** Owns the note rail, including requests that must end when it closes. */
export function mountNotes(
  root: HTMLElement,
  source: () => string | undefined,
  signal: AbortSignal,
) {
  const panel = root.querySelector<HTMLElement>('.flashcards-notes')
  const body = root.querySelector<HTMLElement>('.flashcards-notes-body')
  const title = root.querySelector<HTMLAnchorElement>('.flashcards-notes-title')
  const status = root.querySelector<HTMLElement>('.flashcards-notes-status')
  const button = root.querySelector<HTMLButtonElement>('.fc-notes')
  const french = root.dataset.lang === 'fr'
  const cache = new Map<string, Note>()
  let open = false
  let selected = ''
  let rendered = ''
  let request: AbortController | undefined

  const cancel = () => {
    request?.abort()
    request = undefined
    if (status) setStatusText(status, '')
    body?.removeAttribute('aria-busy')
  }
  const sync = async () => {
    const slug = source()
    if (!open || !slug || !body || !title || !status || signal.aborted) return
    if (slug === selected && (request || slug === rendered)) return
    cancel()
    selected = slug
    title.href = `/${slug}`
    title.dataset.slug = slug
    const sourceTitle =
      slug === root.dataset.source
        ? root.querySelector('.flashcards-card-head h1')?.textContent?.trim()
        : undefined
    title.textContent = cache.get(slug)?.title ?? sourceTitle ?? slug.split('/').at(-1) ?? slug
    const current = new AbortController()
    request = current
    setStatusText(status, french ? 'chargement de la note…' : 'loading note…')
    body.setAttribute('aria-busy', 'true')
    body.replaceChildren()
    try {
      let note = cache.get(slug)
      if (!note) {
        const response = await fetch(`/${slug}`, {
          headers: { Accept: 'text/html' },
          signal: current.signal,
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const doc = new DOMParser().parseFromString(await response.text(), 'text/html')
        normalizeRelativeURLs(doc, new URL(response.url))
        const article = doc.querySelector('.page-content article')
        if (!article) throw new Error('Note content unavailable')
        for (const element of article.querySelectorAll('script')) element.remove()
        const fragment = document.createDocumentFragment()
        fragment.append(document.importNode(article, true))
        note = {
          title: doc.querySelector('h1.article-title')?.textContent?.trim() || slug,
          body: fragment,
        }
      }
      if (current.signal.aborted || signal.aborted || request !== current) return
      cache.set(slug, note)
      rendered = slug
      title.textContent = note.title
      body.replaceChildren(note.body.cloneNode(true))
      body.scrollTop = 0
    } catch {
      if (current.signal.aborted || signal.aborted || request !== current) return
      setStatusText(status, french ? 'note indisponible' : 'note unavailable')
      const error = document.createElement('p')
      error.className = 'flashcards-notes-empty'
      error.textContent = french
        ? 'Fermez puis rouvrez les notes pour réessayer.'
        : 'Close and reopen notes to retry, or open the source above.'
      body.replaceChildren(error)
      rendered = ''
    } finally {
      if (request === current) {
        request = undefined
        body.removeAttribute('aria-busy')
        if (rendered === slug) setStatusText(status, '')
      }
    }
  }
  const setOpen = (value: boolean) => {
    open = value
    root.dataset.notes = value ? 'open' : 'closed'
    panel?.toggleAttribute('hidden', !value)
    button?.setAttribute('aria-pressed', String(value))
    if (value) void sync()
    else cancel()
  }
  signal.addEventListener('abort', cancel, { once: true })
  return { sync, setOpen, toggle: () => setOpen(!open) }
}
