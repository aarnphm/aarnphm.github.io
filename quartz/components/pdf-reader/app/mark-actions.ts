import { useCallback } from 'preact/hooks'
import type { PdfMarkKind, PdfMarkTarget } from '../../../util/pdf-marks'
import type { MarkBook, MarkEntry } from '../book'
import type { PageSelection } from '../geometry'
import type { PdfViewer } from '../viewer'
import { boxQuad } from '../geometry'
import { copyText, toast } from './utils'

interface ReaderRef<T> {
  current: T | null
}

interface ActiveSelection extends PageSelection {
  range: Range
}

interface MarkActionOptions {
  slug: string
  title: string
  label: (page: number) => string
  viewerRef: ReaderRef<PdfViewer>
  book: MarkBook
  selection: ActiveSelection | null
  setSelection: (selection: ActiveSelection | null) => void
  activate: (id: string | null, options?: { scroll?: boolean; focus?: boolean }) => void
  page: number
  entries: MarkEntry[]
  activeId: string | null
  setActiveId: (update: (current: string | null) => string | null) => void
  readPath: string
}

export function useMarkActions({
  slug,
  title,
  label,
  viewerRef,
  book,
  selection,
  setSelection,
  activate,
  page,
  entries,
  activeId,
  setActiveId,
  readPath,
}: MarkActionOptions) {
  const quoteText = useCallback(
    (quote: string | null, number: number, id?: string) => {
      const anchor = id ? `#^${id}` : `#page=${number}`
      const link = `[[${slug}${anchor}|${title}, p. ${label(number)}]]`
      if (!quote) return link
      return `${quote
        .split('\n')
        .map(line => `> ${line}`)
        .join('\n')}\n> ${link}`
    },
    [slug, title, label],
  )

  const markFromSelection = useCallback(
    (kind: PdfMarkKind, focus: boolean) => {
      const viewer = viewerRef.current
      if (!selection || !viewer || !book.canWrite) return
      const base = viewer.pages[selection.page - 1]?.base
      if (!base) return
      const target: PdfMarkTarget = {
        type: 'text',
        page: selection.page,
        quads: selection.boxes.map(box => boxQuad(box, base)),
        quote: selection.quote,
      }
      const entry = book.create(kind, target)
      document.getSelection()?.removeAllRanges()
      setSelection(null)
      activate(entry.mark.id, { focus })
      if (selection.clipped)
        toast(`Marks stay on one page, so this one covers p. ${label(selection.page)}.`)
    },
    [selection, book, activate, label],
  )

  const onBubble = useCallback(
    (action: PdfMarkKind | 'note' | 'quote') => {
      if (action === 'quote') {
        if (!selection) return
        void copyText(
          quoteText(selection.quote.exact, selection.page),
          'Quote copied as a garden link.',
        )
        return
      }
      if (action === 'note') markFromSelection('mark', true)
      else markFromSelection(action, action !== 'mark')
    },
    [selection, quoteText, markFromSelection],
  )

  const pageNote = useCallback(() => {
    if (!book.canWrite) return
    const entry = book.create('mark', { type: 'page', page })
    activate(entry.mark.id, { focus: true })
  }, [book, page, activate])

  const removeMark = useCallback(
    (id: string) => {
      book.remove(id)
      setActiveId(current => (current === id ? null : current))
      toast('Mark deleted.', { label: 'undo', onClick: () => book.undo() }, 6000)
    },
    [book],
  )

  const step = useCallback(
    (direction: 1 | -1) => {
      if (entries.length === 0) return
      const index = entries.findIndex(entry => entry.mark.id === activeId)
      let next: MarkEntry | undefined
      if (index >= 0) next = entries[index + direction]
      else if (direction > 0) next = entries.find(entry => entry.mark.target.page >= page)
      else next = [...entries].reverse().find(entry => entry.mark.target.page <= page)
      if (next) activate(next.mark.id, { scroll: true })
    },
    [entries, activeId, page, activate],
  )

  const copyMarkQuote = async (id: string): Promise<boolean> => {
    const entry = book.entries.get(id)
    if (!entry) return false
    const quote = entry.mark.target.type === 'text' ? entry.mark.target.quote.exact : null
    return copyText(quoteText(quote, entry.mark.target.page, id), 'Quote copied as a garden link.')
  }

  const copyLink = (id: string) =>
    copyText(`${location.origin}${encodeURI(readPath)}#^${id}`, 'Link copied.')

  return {
    quoteText,
    markFromSelection,
    onBubble,
    pageNote,
    removeMark,
    step,
    copyMarkQuote,
    copyLink,
  }
}
