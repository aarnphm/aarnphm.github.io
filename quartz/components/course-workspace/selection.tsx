import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'
import { useLayoutEffect, useRef } from 'preact/hooks'
import type { PdfMarkKind } from '../../util/pdf-marks'
import { kindMeta } from '../pdf-reader/parts'

export const noteKinds: PdfMarkKind[] = ['mark', 'question', 'contra']
export type SelectionAction = PdfMarkKind | 'note' | 'scratchpad'

export function SelectionActions({
  range,
  onAction,
}: {
  range: Range
  onAction(action: SelectionAction): void
}) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const floating = ref.current
    if (!floating) return
    let disposed = false
    const reference = {
      // Union of the painted line boxes, so the toolbar centers on the text. A range that ends at
      // offset 0 of the next block (triple-click) also reports an empty rect there, which is skipped.
      getBoundingClientRect: () => {
        const rects = [...range.getClientRects()].filter(rect => rect.width > 0)
        if (!rects.length) return range.getBoundingClientRect()
        const left = Math.min(...rects.map(rect => rect.left))
        const top = Math.min(...rects.map(rect => rect.top))
        const right = Math.max(...rects.map(rect => rect.right))
        const bottom = Math.max(...rects.map(rect => rect.bottom))
        return new DOMRect(left, top, right - left, bottom - top)
      },
      contextElement: range.endContainer.parentElement ?? undefined,
    }
    const cleanup = autoUpdate(reference, floating, async () => {
      const { x, y } = await computePosition(reference, floating, {
        placement: 'bottom',
        strategy: 'fixed',
        middleware: [offset(6), flip({ padding: 12 }), shift({ padding: 12, crossAxis: true })],
      })
      if (!disposed)
        Object.assign(floating.style, { left: `${x}px`, top: `${y}px`, visibility: 'visible' })
    })
    return () => {
      disposed = true
      cleanup()
    }
  }, [range])
  return (
    <div
      ref={ref}
      class="course-selection-actions"
      role="toolbar"
      aria-label="Lecture selection"
      onPointerDown={event => event.preventDefault()}
    >
      {noteKinds.map(kind => (
        <button type="button" onClick={() => onAction(kind)}>
          {kindMeta[kind].glyph} {kindMeta[kind].label}
        </button>
      ))}
      <button type="button" onClick={() => onAction('note')}>
        note
      </button>
      <button type="button" onClick={() => onAction('scratchpad')}>
        → scratchpad
      </button>
    </div>
  )
}
