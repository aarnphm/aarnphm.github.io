import {
  noteLimit,
  loadNotes,
  removeNote,
  saveNote,
  type LectureNote,
} from '../course-workspace/store'
// Reader notes in the /fr margin. The notes are the lecture workspace's: the same browser store,
// the same Markdown editor, the same rendered body. Loaded on demand as the LazyScripts bundle
// `fr-notes.js`, so CodeMirror and marked stay out of the page chunk.
import { renderBody } from '../pdf-reader/body'
import { MarkdownEditor } from './markdown-editor'

const MONTHS = [
  'janv.',
  'févr.',
  'mars',
  'avr.',
  'mai',
  'juin',
  'juil.',
  'août',
  'sept.',
  'oct.',
  'nov.',
  'déc.',
]

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string) => {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const button = (label: string, className = 'fr-btn') => {
  const node = el('button', className, label)
  node.type = 'button'
  return node
}

const dayLabel = (at: number) => {
  const d = new Date(at)
  return `${d.getDate()} ${MONTHS[d.getMonth()]}`
}

export function mountFrNotes(
  root: HTMLElement,
  slug: string,
  signal: AbortSignal,
  relayout: () => void,
) {
  const section = root.dataset.section ?? ''
  const list = root.querySelector<HTMLElement>('[data-fr-note-list]')
  const found = root.querySelector<HTMLButtonElement>('[data-fr-act="add"]')
  if (!list || !found) return
  const add = found
  let notes: LectureNote[] = []
  const read = () => {
    try {
      notes = loadNotes(slug).filter(note => note.deletedAt === null && note.heading === section)
    } catch {
      notes = []
    }
  }
  let session: { view: MarkdownEditor; host: HTMLElement } | undefined
  let draft = ''
  let editing: LectureNote | undefined

  const paint = () => {
    list.replaceChildren(
      ...notes.map(note => {
        const row = el('span', 'fr-note')
        const body = el('span', 'course-draft-preview')
        body.innerHTML = renderBody(note.body)
        const meta = el('span', 'fr-meta', `${dayLabel(note.createdAt)} · privé`)
        const edit = button('modifier', 'fr-btn fr-link')
        edit.addEventListener('click', () => open(note), { signal })
        const card = button('carte', 'fr-btn fr-link')
        card.addEventListener(
          'click',
          () => {
            const source = `Q: ${note.body}\nA: \nN: [[${slug}#${section}]]\n`
            void navigator.clipboard?.writeText(source).then(() => {
              card.textContent = 'copié'
              setTimeout(() => (card.textContent = 'carte'), 1500)
            })
          },
          { signal },
        )
        const drop = button('retirer', 'fr-btn fr-link')
        drop.addEventListener(
          'click',
          () => {
            removeNote(slug, note.id)
            read()
            paint()
          },
          { signal },
        )
        meta.append(' ', edit, ' ', card, ' ', drop)
        row.append(body, meta)
        return row
      }),
    )
    relayout()
  }

  const close = () => {
    session?.view.destroy()
    session?.host.remove()
    session = undefined
    editing = undefined
    add.hidden = false
    relayout()
  }

  const commit = () => {
    const body = draft.trim()
    if (body) {
      try {
        saveNote(
          slug,
          editing
            ? { ...editing, body }
            : {
                id: crypto.randomUUID(),
                kind: 'mark',
                quote: null,
                heading: section,
                body,
                revision: 0,
                createdAt: Date.now(),
                deletedAt: null,
              },
        )
      } catch {
        /* quota or a newer copy in another tab: the editor text stays on screen */
        return
      }
      read()
      paint()
    }
    close()
  }

  function open(note?: LectureNote) {
    close()
    editing = note
    draft = note?.body ?? ''
    const host = el('span', 'fr-editor')
    const field = el('span', 'course-draft-editor')
    const save = button('ok')
    save.addEventListener('click', commit, { signal })
    host.append(field, save)
    add.hidden = true
    add.after(host)
    const view = new MarkdownEditor({
      parent: field,
      initialContent: draft,
      ariaLabel: 'nouvelle note',
      placeholder: '',
      maxLength: noteLimit,
      changeDebounceMs: 120,
      onChange: body => (draft = body),
      onCancel: close,
      onSubmit: commit,
    })
    session = { view, host }
    view.focus()
    relayout()
  }

  add.addEventListener('click', () => open(), { signal })
  signal.addEventListener('abort', () => close(), { once: true })
  read()
  paint()
}
