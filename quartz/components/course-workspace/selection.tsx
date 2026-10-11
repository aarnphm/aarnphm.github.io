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
      getBoundingClientRect: () => {
        const rects = range.getClientRects()
        return rects[rects.length - 1] ?? range.getBoundingClientRect()
      },
      contextElement: range.endContainer.parentElement ?? undefined,
    }
    const cleanup = autoUpdate(reference, floating, async () => {
      const { x, y } = await computePosition(reference, floating, {
        placement: 'bottom-start',
        strategy: 'fixed',
        middleware: [offset(6), flip({ padding: 12 }), shift({ padding: 12 })],
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
