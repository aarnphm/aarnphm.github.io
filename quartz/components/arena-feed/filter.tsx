import { useEffect, useId, useLayoutEffect, useRef, useState } from 'preact/hooks'

export function ReaderFilter<Value extends string>({
  label,
  hideLabel = false,
  options,
  value,
  onChange,
}: {
  label: string
  hideLabel?: boolean
  options: readonly { value: Value; label: string }[]
  value: Value
  onChange: (value: Value) => void
}) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)

  const show = () => {
    setActive(options.findIndex(option => option.value === value))
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
      <span id={`${id}-label`} hidden={hideLabel}>
        {label}
      </span>
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
          <span id={`${id}-value`}>{options.find(option => option.value === value)?.label}</span>
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" focusable="false">
            <path d="m4 6 4 4 4-4" stroke="currentColor" stroke-width="1.25" />
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
                  ? options.length - 1
                  : event.key === 'ArrowDown'
                    ? Math.min(options.length - 1, active + 1)
                    : event.key === 'ArrowUp'
                      ? Math.max(0, active - 1)
                      : event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey
                        ? options.findIndex(option =>
                            option.label.toLowerCase().startsWith(event.key.toLowerCase()),
                          )
                        : -1
            if (next < 0) return
            event.preventDefault()
            setActive(next)
          }}
        >
          {options.map((option, index) => (
            <button
              key={option.value}
              type="button"
              role="option"
              aria-selected={option.value === value}
              tabIndex={index === active ? 0 : -1}
              onFocus={() => setActive(index)}
              onClick={() => {
                onChange(option.value)
                close()
              }}
            >
              <svg
                class="arena-reader-filter-check"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
                focusable="false"
              >
                <path d="m3 8 3 3 7-7" stroke="currentColor" stroke-width="1.25" />
              </svg>
              <span>{option.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
