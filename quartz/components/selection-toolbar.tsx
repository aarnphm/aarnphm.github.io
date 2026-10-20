import type { Placement } from '@floating-ui/dom'
import type { ComponentChildren } from 'preact'
import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'
import { useLayoutEffect, useRef } from 'preact/hooks'

export function selectionReference(range: Range) {
  return {
    getBoundingClientRect: () => {
      const rects = [...range.getClientRects()].filter(rect => rect.width > 0)
      if (!rects.length) return range.getBoundingClientRect()
      const left = Math.min(...rects.map(rect => rect.left))
      const top = Math.min(...rects.map(rect => rect.top))
      const right = Math.max(...rects.map(rect => rect.right))
      const bottom = Math.max(...rects.map(rect => rect.bottom))
      return new DOMRect(left, top, right - left, bottom - top)
    },
    contextElement:
      range.endContainer instanceof Element
        ? range.endContainer
        : (range.endContainer.parentElement ?? undefined),
  }
}

export function SelectionToolbar({
  range,
  label,
  placement = 'bottom',
  children,
}: {
  range: Range
  label: string
  placement?: Placement
  children: ComponentChildren
}) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const floating = ref.current
    if (!floating) return
    let disposed = false
    const reference = selectionReference(range)
    const cleanup = autoUpdate(reference, floating, async () => {
      const { x, y } = await computePosition(reference, floating, {
        placement,
        strategy: 'fixed',
        middleware: [offset(6), flip({ padding: 12 }), shift({ padding: 12, crossAxis: true })],
      })
      if (!disposed) {
        Object.assign(floating.style, { left: `${x}px`, top: `${y}px`, visibility: 'visible' })
      }
    })
    return () => {
      disposed = true
      cleanup()
    }
  }, [range, placement])
  return (
    <div
      ref={ref}
      class="selection-toolbar"
      role="toolbar"
      aria-label={label}
      onPointerDown={event => event.preventDefault()}
    >
      {children}
    </div>
  )
}
