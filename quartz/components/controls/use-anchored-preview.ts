import type { TargetedFocusEvent, TargetedKeyboardEvent, TargetedPointerEvent } from 'preact'
import { autoUpdate, computePosition, flip, offset, shift, size } from '@floating-ui/dom'
import { useLayoutEffect, useRef, useState } from 'preact/hooks'

export interface AnchoredPreview<T> {
  item: T
  anchor: HTMLElement
}

const HIDE_DELAY_MS = 180

/**
 * A preview panel for the item under the mouse or keyboard focus, in a `popover="manual"`
 * element placed under its anchor. Escape or a click on an anchor dismisses it; the preview
 * stays dismissed until the pointer moves or focus goes to another anchor, so closing it
 * cannot reopen it under a stationary pointer.
 */
export const useAnchoredPreview = <T>({
  canShow = () => true,
  resetKeys,
  resetAnchor,
}: {
  canShow?: (item: T) => boolean
  /** A change in any key closes the preview and marks it dismissed. */
  resetKeys: unknown[]
  /** On reset, the anchor that must not reopen the preview on focus; `undefined` keeps the last one. */
  resetAnchor?: (popover: HTMLElement | null) => HTMLElement | null | undefined
}) => {
  const [preview, setPreview] = useState<AnchoredPreview<T> | null>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout>>()
  const dismissed = useRef(false)
  const dismissedAnchor = useRef<HTMLElement | null>(null)
  const cancelHide = () => clearTimeout(hideTimer.current)
  const dismiss = () => {
    cancelHide()
    if (popoverRef.current?.matches(':popover-open')) popoverRef.current.hidePopover()
    setPreview(null)
  }
  const scheduleHide = () => {
    cancelHide()
    hideTimer.current = setTimeout(() => setPreview(null), HIDE_DELAY_MS)
  }
  const show = (item: T, anchor: HTMLElement) => {
    cancelHide()
    if (canShow(item) && preview?.anchor !== anchor) setPreview({ item, anchor })
  }
  const dismissFrom = (anchor: HTMLElement) => {
    dismissed.current = true
    dismissedAnchor.current = preview?.anchor ?? anchor
    dismiss()
  }

  useLayoutEffect(() => {
    dismissed.current = true
    const anchor = resetAnchor?.(popoverRef.current)
    if (anchor !== undefined) dismissedAnchor.current = anchor
    setPreview(null)
  }, resetKeys)
  useLayoutEffect(() => () => clearTimeout(hideTimer.current), [])
  useLayoutEffect(() => {
    const panel = popoverRef.current
    if (!preview || !panel) return
    let disposed = false
    panel.showPopover()
    const update = async () => {
      if (!preview.anchor.getClientRects().length) return setPreview(null)
      const { x, y } = await computePosition(preview.anchor, panel, {
        strategy: 'fixed',
        placement: 'bottom-start',
        middleware: [
          offset(6),
          flip({ padding: 8 }),
          shift({ padding: 8 }),
          size({
            padding: 8,
            apply: ({ availableHeight }) => {
              panel.style.maxHeight = `${Math.max(0, availableHeight)}px`
            },
          }),
        ],
      })
      if (disposed) return
      panel.style.left = `${x}px`
      panel.style.top = `${y}px`
    }
    const cleanup = autoUpdate(preview.anchor, panel, update)
    return () => {
      disposed = true
      cleanup()
      if (panel.matches(':popover-open')) panel.hidePopover()
    }
  }, [preview])

  const anchorProps = (item: T) => ({
    onPointerEnter: (event: TargetedPointerEvent<HTMLElement>) => {
      if (event.pointerType === 'mouse' && !dismissed.current) show(item, event.currentTarget)
    },
    onPointerMove: (event: TargetedPointerEvent<HTMLElement>) => {
      // Closing a preview can expose another anchor under the stationary pointer.
      if (event.pointerType !== 'mouse' || !dismissed.current) return
      dismissed.current = false
      show(item, event.currentTarget)
    },
    onPointerLeave: scheduleHide,
    onFocus: (event: TargetedFocusEvent<HTMLElement>) => {
      if (dismissed.current && dismissedAnchor.current === event.currentTarget) return
      dismissed.current = false
      show(item, event.currentTarget)
    },
    onBlur: (event: TargetedFocusEvent<HTMLElement>) => {
      if (
        !(event.relatedTarget instanceof Node && popoverRef.current?.contains(event.relatedTarget))
      )
        dismiss()
    },
  })

  const popoverProps = {
    onPointerEnter: cancelHide,
    onPointerLeave: scheduleHide,
    onFocusIn: cancelHide,
    onFocusOut: (event: TargetedFocusEvent<HTMLElement>) => {
      const next = event.relatedTarget
      if (
        !(
          next instanceof Node &&
          (popoverRef.current?.contains(next) || preview?.anchor.contains(next))
        )
      )
        dismiss()
    },
  }

  const onKeyDown = (event: TargetedKeyboardEvent<HTMLElement>) => {
    if (event.key !== 'Escape' || !preview) return
    event.preventDefault()
    event.stopPropagation()
    const anchor = preview.anchor
    dismissFrom(anchor)
    anchor.focus({ preventScroll: true })
  }

  return { preview, popoverRef, anchorProps, popoverProps, onKeyDown, dismissFrom }
}
