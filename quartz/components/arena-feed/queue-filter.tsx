import { useEffect, useId, useLayoutEffect, useRef, useState } from 'preact/hooks'
import type { FeedFilter } from './model'

const filters: { value: FeedFilter; label: string }[] = [
  { value: 'unread', label: 'unread' },
  { value: 'read', label: 'read' },
  { value: 'all', label: 'all saved links' },
]

export function QueueFilter({
  value,
  onChange,
}: {
  value: FeedFilter
  onChange: (value: FeedFilter) => void
}) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)

  const show = () => {
    setActive(filters.findIndex(filter => filter.value === value))
    setOpen(true)
  }
  const close = () => {
    setOpen(false)
    trigger.current?.focus({ preventScroll: true })
  }

  useLayoutEffect(() => {
    if (open) root.current?.querySelectorAll<HTMLButtonElement>('[role="option"]')[active]?.focus()
  }, [open, active])

  useEffect(() => {
    if (!open) return
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('pointerdown', dismiss)
    return () => document.removeEventListener('pointerdown', dismiss)
  }, [open])

  return (
    <div
      class="arena-reader-filter"
      ref={root}
      onFocusOut={event => {
        if (event.relatedTarget instanceof Node && root.current?.contains(event.relatedTarget))
          return
        setOpen(false)
      }}
    >
      <span id={`${id}-label`}>show</span>
      <div class="arena-reader-filter-picker">
        <button
          ref={trigger}
          class="arena-reader-filter-trigger"
          type="button"
          aria-labelledby={`${id}-label ${id}-value`}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={`${id}-menu`}
          onClick={() => (open ? setOpen(false) : show())}
          onKeyDown={event => {
            if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
            event.preventDefault()
            show()
          }}
        >
          <span id={`${id}-value`}>{filters.find(filter => filter.value === value)?.label}</span>
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" focusable="false">
            <path d="m4 6 4 4 4-4" stroke="currentColor" stroke-width="1.4" />
          </svg>
        </button>
        <div
          id={`${id}-menu`}
          class="arena-reader-filter-menu"
          role="listbox"
          aria-labelledby={`${id}-label`}
          hidden={!open}
          onKeyDown={event => {
            if (event.key === 'Escape') {
              event.preventDefault()
              event.stopPropagation()
              close()
              return
            }
            const next =
              event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? filters.length - 1
                  : event.key === 'ArrowDown'
                    ? Math.min(filters.length - 1, active + 1)
                    : event.key === 'ArrowUp'
                      ? Math.max(0, active - 1)
                      : event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey
                        ? filters.findIndex(filter =>
                            filter.label.toLowerCase().startsWith(event.key.toLowerCase()),
                          )
                        : -1
            if (next < 0) return
            event.preventDefault()
            setActive(next)
          }}
        >
          {filters.map((filter, index) => (
            <button
              key={filter.value}
              type="button"
              role="option"
              aria-selected={filter.value === value}
              tabIndex={index === active ? 0 : -1}
              onFocus={() => setActive(index)}
              onClick={() => {
                onChange(filter.value)
                close()
              }}
            >
              <span class="arena-reader-filter-check" aria-hidden="true">
                ✓
              </span>
              <span>{filter.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
