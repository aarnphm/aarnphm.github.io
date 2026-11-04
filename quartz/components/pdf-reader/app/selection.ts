import { useEffect } from 'preact/hooks'
import { selectionOnPage, type PageSelection } from '../geometry'

export interface ActiveSelection extends PageSelection {
  range: Range
}

interface SelectionRef {
  current: HTMLDivElement | null
}

export function useReaderSelection(
  columnRef: SelectionRef,
  setSelection: (selection: ActiveSelection | null) => void,
) {
  useEffect(() => {
    let timer = 0
    const onChange = () => {
      window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        const current = document.getSelection()
        const column = columnRef.current
        if (!current || current.isCollapsed || current.rangeCount === 0 || !column) {
          setSelection(null)
          return
        }
        const range = current.getRangeAt(0)
        if (!column.contains(range.commonAncestorContainer)) {
          setSelection(null)
          return
        }
        const resolved = selectionOnPage(range)
        setSelection(resolved ? { ...resolved, range: range.cloneRange() } : null)
      }, 90)
    }
    const release = () => {
      for (const layer of columnRef.current?.querySelectorAll('.pdf-reader-text.selecting') ?? []) {
        layer.classList.remove('selecting')
      }
    }
    document.addEventListener('selectionchange', onChange)
    window.addEventListener('pointerup', release)
    return () => {
      window.clearTimeout(timer)
      document.removeEventListener('selectionchange', onChange)
      window.removeEventListener('pointerup', release)
    }
  }, [])
}
