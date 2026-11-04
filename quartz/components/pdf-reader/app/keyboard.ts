import { useEffect } from 'preact/hooks'
import type { PdfMarkKind } from '../../../util/pdf-marks'
import type { MarkBook } from '../book'
import type { PageSelection } from '../geometry'
import type { PdfViewer } from '../viewer'
import { followCourseKey, isEditable } from './utils'

interface ReaderRef<T> {
  current: T | null
}

interface ReaderSelection extends PageSelection {
  range: Range
}

interface ReaderKeyboardOptions {
  activeId: string | null
  compact: boolean
  label: (page: number) => string
  page: number
  tool: 'select' | 'region'
  selection: ReaderSelection | null
  book: MarkBook
  viewerRef: ReaderRef<PdfViewer>
  keysRef: ReaderRef<HTMLDialogElement>
  activate: (id: string | null) => void
  step: (direction: 1 | -1) => void
  onBubble: (action: PdfMarkKind | 'note' | 'quote') => void
  copyMarkQuote: (id: string) => Promise<boolean>
  markFromSelection: (kind: PdfMarkKind, focus: boolean) => void
  pageNote: () => void
  setTool: (
    tool: 'select' | 'region' | ((current: 'select' | 'region') => 'select' | 'region'),
  ) => void
  setSelection: (selection: ReaderSelection | null) => void
  removeMark: (id: string) => void
}

export function useReaderKeyboard({
  activeId,
  compact,
  label,
  page,
  tool,
  selection,
  book,
  viewerRef,
  keysRef,
  activate,
  step,
  onBubble,
  copyMarkQuote,
  markFromSelection,
  pageNote,
  setTool,
  setSelection,
  removeMark,
}: ReaderKeyboardOptions) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (isEditable(event.target) || event.altKey) return
      if (document.querySelector('dialog[open]:not(.pdf-reader-sheet)')) return
      const viewer = viewerRef.current
      const mod = event.metaKey || event.ctrlKey
      let handled = true
      if (mod && event.key.toLowerCase() === 'z' && !event.shiftKey && book.canUndo) {
        const restored = book.undo()
        if (restored) activate(restored.mark.id)
      } else if (mod) {
        handled = false
      } else if (event.key === 'Escape') {
        const sheet = document.querySelector<HTMLElement>('[popover]:popover-open')
        if (sheet) sheet.hidePopover()
        else if (tool === 'region') setTool('select')
        else if (selection) {
          document.getSelection()?.removeAllRanges()
          setSelection(null)
        } else if (activeId) activate(null)
        else handled = false
      } else if (event.key === '?') {
        keysRef.current?.showModal()
      } else if (event.key === '[' || event.key === ']') {
        step(event.key === ']' ? 1 : -1)
      } else if (event.key === '<' || event.key === '>' || event.key === 'u') {
        handled = followCourseKey(event.key === '<' ? 'prev' : event.key === '>' ? 'next' : 'up')
      } else if (event.key === '+' || event.key === '=') {
        viewer?.zoomBy(1.1)
      } else if (event.key === '-') {
        viewer?.zoomBy(1 / 1.1)
      } else if (event.key === '0') {
        viewer?.setZoom('width')
      } else if (event.key === 'c' && (selection || activeId)) {
        if (selection) onBubble('quote')
        else if (activeId) void copyMarkQuote(activeId)
      } else if (!book.canWrite) {
        handled = false
      } else if (event.key === 'a' || event.key === 'q' || event.key === 'x') {
        const kind: PdfMarkKind =
          event.key === 'a' ? 'mark' : event.key === 'q' ? 'question' : 'contra'
        if (selection) markFromSelection(kind, kind !== 'mark')
        else if (activeId) book.update(activeId, { kind }, true)
        else handled = false
      } else if (event.key === 'n') {
        if (selection) markFromSelection('mark', true)
        else pageNote()
      } else if (event.key === 'r') {
        setTool(current => (current === 'region' ? 'select' : 'region'))
      } else if ((event.key === 'Delete' || event.key === 'Backspace') && activeId) {
        removeMark(activeId)
      } else {
        handled = false
      }
      if (handled) {
        event.preventDefault()
        event.stopImmediatePropagation()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [activeId, compact, label, page, tool])
}
