import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'
import { useId, useLayoutEffect, useRef, useState } from 'preact/hooks'

export function DeleteNote({ disabled, onDelete }: { disabled: boolean; onDelete: () => void }) {
  const id = useId()
  const trigger = useRef<HTMLButtonElement>(null)
  const popover = useRef<HTMLDialogElement>(null)
  const cancel = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)

  useLayoutEffect(() => {
    const button = trigger.current
    const dialog = popover.current
    if (!open || !button || !dialog) return
    let mounted = true
    let focused = false
    const cleanup = autoUpdate(button, dialog, async () => {
      const { x, y } = await computePosition(button, dialog, {
        placement: 'top-end',
        strategy: 'fixed',
        middleware: [offset(8), flip(), shift({ padding: 8 })],
      })
      if (!mounted || !dialog.matches(':popover-open')) return
      Object.assign(dialog.style, { left: `${x}px`, top: `${y}px`, visibility: 'visible' })
      if (!focused) {
        focused = true
        cancel.current?.focus({ preventScroll: true })
      }
    })
    return () => {
      mounted = false
      cleanup()
    }
  }, [open])

  return (
    <>
      <button
        ref={trigger}
        type="button"
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        popovertarget={id}
      >
        delete
      </button>
      <dialog
        ref={popover}
        id={id}
        class="arena-note-delete"
        popover="auto"
        aria-labelledby={`${id}-title`}
        aria-describedby={`${id}-description`}
        onBeforeToggle={event => {
          if (event.newState === 'open') event.currentTarget.style.visibility = 'hidden'
        }}
        onToggle={event => setOpen(event.newState === 'open')}
        onKeyDown={event => {
          if (event.key !== 'Escape') return
          event.preventDefault()
          event.stopPropagation()
          event.currentTarget.hidePopover()
        }}
        onFocusOut={event => {
          if (
            event.relatedTarget instanceof Node &&
            !event.currentTarget.contains(event.relatedTarget)
          )
            event.currentTarget.hidePopover()
        }}
      >
        <h3 id={`${id}-title`}>delete this note?</h3>
        <p id={`${id}-description`}>This removes the note from your reader.</p>
        <div class="arena-note-delete-actions">
          <button
            ref={cancel}
            type="button"
            popovertarget={id}
            popovertargetaction="hide"
            autoFocus
          >
            cancel
          </button>
          <button
            type="button"
            onClick={() => {
              popover.current?.hidePopover()
              onDelete()
            }}
          >
            delete note
          </button>
        </div>
      </dialog>
    </>
  )
}
