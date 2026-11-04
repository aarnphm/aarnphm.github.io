import type { MarkBook } from '../book'
import type { PageBox } from '../geometry'
import type { PdfViewer } from '../viewer'
import { boxRect, hitTest } from '../geometry'

interface ReaderRef<T> {
  current: T | null
}

interface PointerHandlerOptions {
  tool: 'select' | 'region'
  viewerRef: ReaderRef<PdfViewer>
  book: MarkBook
  activate: (id: string | null, options?: { scroll?: boolean; focus?: boolean }) => void
  setTool: (tool: 'select' | 'region') => void
  boxesById: Map<string, { page: number; boxes: PageBox[] }>
  activeId: string | null
}

export function createPointerHandlers({
  tool,
  viewerRef,
  book,
  activate,
  setTool,
  boxesById,
  activeId,
}: PointerHandlerOptions) {
  const onPointerDown = (event: PointerEvent) => {
    const target = event.target as Element
    const pageElement = target.closest<HTMLElement>('[data-page-number]')
    if (!pageElement) return
    if (tool !== 'region') {
      pageElement.querySelector('.pdf-reader-text')?.classList.add('selecting')
      return
    }
    const viewer = viewerRef.current
    const number = Number(pageElement.dataset.pageNumber)
    const base = viewer?.pages[number - 1]?.base
    if (!base || event.button !== 0) return
    event.preventDefault()
    const rect = pageElement.getBoundingClientRect()
    const start = {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    }
    const draft = document.createElement('div')
    draft.className = 'pdf-reader-region-draft'
    pageElement.append(draft)
    let box: PageBox = { left: start.x, top: start.y, width: 0, height: 0 }
    const move = (moveEvent: PointerEvent) => {
      const x = Math.min(1, Math.max(0, (moveEvent.clientX - rect.left) / rect.width))
      const y = Math.min(1, Math.max(0, (moveEvent.clientY - rect.top) / rect.height))
      box = {
        left: Math.min(start.x, x),
        top: Math.min(start.y, y),
        width: Math.abs(x - start.x),
        height: Math.abs(y - start.y),
      }
      Object.assign(draft.style, {
        left: `${box.left * 100}%`,
        top: `${box.top * 100}%`,
        width: `${box.width * 100}%`,
        height: `${box.height * 100}%`,
      })
    }
    const finish = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', finish)
      window.removeEventListener('pointercancel', finish)
      draft.remove()
      setTool('select')
      if (box.width < 0.01 || box.height < 0.005) return
      const entry = book.create('mark', { type: 'region', page: number, rect: boxRect(box, base) })
      activate(entry.mark.id, { focus: true })
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', finish)
    window.addEventListener('pointercancel', finish)
  }

  const onPointerUp = (event: PointerEvent) => {
    if (tool === 'region' || event.button !== 0) return
    const current = document.getSelection()
    if (current && !current.isCollapsed) return
    const target = event.target as Element
    if (target.closest('a')) return
    const pageElement = target.closest<HTMLElement>('[data-page-number]')
    if (!pageElement) return
    const number = Number(pageElement.dataset.pageNumber)
    const rect = pageElement.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    const onPage = new Map<string, PageBox[]>()
    for (const [id, placed] of boxesById) if (placed.page === number) onPage.set(id, placed.boxes)
    const hits = hitTest(onPage, x, y)
    if (hits.length > 0) activate(hits[hits.length - 1])
    else if (activeId) activate(null)
  }

  return { onPointerDown, onPointerUp }
}
