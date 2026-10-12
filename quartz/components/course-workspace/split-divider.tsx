import { useLayoutEffect, useRef, useState } from 'preact/hooks'

// v2: the default moved from an even split to the page reading column, so v1 halves are dropped.
const preferenceKey = 'course-workspace:split:v2'
// The reading column spans 3 of the 7 tracks the split covers (see course-workspace.scss).
const defaultShare = (3 / 7) * 100

function savedShare(): number {
  try {
    const value = localStorage.getItem(preferenceKey)
    const share = value === null ? defaultShare : Number(value)
    return Number.isFinite(share) && share > 0 && share < 100 ? share : defaultShare
  } catch {
    return defaultShare
  }
}

interface SplitProps {
  container: HTMLElement
  article: HTMLElement
  workspace: HTMLElement
  enabled: boolean
}

export function SplitDivider({ container, article, workspace, enabled }: SplitProps) {
  const [share, setShare] = useState(savedShare)
  const [limits, setLimits] = useState({ min: 0, max: 100 })
  const preferred = useRef(share)
  const geometry = useRef({ width: 0, min: 0, max: 100 })
  const drag = useRef<{ pointer: number; x: number; share: number; width: number } | null>(null)

  const apply = (value: number) => {
    const next = Math.min(geometry.current.max, Math.max(geometry.current.min, value))
    container.style.setProperty('--course-reading-track', `${next}fr`)
    container.style.setProperty('--course-workspace-track', `${100 - next}fr`)
    setShare(next)
    return next
  }
  const save = () => {
    try {
      localStorage.setItem(preferenceKey, String(preferred.current))
    } catch {
      // Resizing remains available when browser preferences cannot be saved.
    }
  }
  const change = (value: number) => {
    preferred.current = apply(value)
    save()
  }

  useLayoutEffect(() => {
    if (!enabled) return
    let lastWidth = -1
    const measure = () => {
      const width = container.getBoundingClientRect().width
      if (Math.abs(width - lastWidth) < 0.1) return
      lastWidth = width
      const available = width - (Number.parseFloat(getComputedStyle(container).columnGap) || 0)
      if (available <= 0) return
      const readingMin = Number.parseFloat(getComputedStyle(article).minWidth) || 0
      const workspaceMin = Number.parseFloat(getComputedStyle(workspace).minWidth) || 0
      const min = Math.min(defaultShare, (readingMin / available) * 100)
      const max = Math.max(defaultShare, (1 - workspaceMin / available) * 100)
      geometry.current = { width: available, min, max }
      setLimits({ min, max })
      apply(preferred.current)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(container)
    return () => {
      observer.disconnect()
      drag.current = null
      delete container.dataset.resizing
      container.style.removeProperty('--course-reading-track')
      container.style.removeProperty('--course-workspace-track')
    }
  }, [container, article, workspace, enabled])

  const move = (event: PointerEvent) => {
    const start = drag.current
    if (!start || start.pointer !== event.pointerId || start.width <= 0) return
    preferred.current = apply(start.share + ((event.clientX - start.x) / start.width) * 100)
  }
  const finish = (event: PointerEvent) => {
    if (drag.current?.pointer !== event.pointerId) return
    drag.current = null
    delete container.dataset.resizing
    save()
  }

  if (!enabled) return null
  return (
    <div
      class="course-workspace-divider"
      role="separator"
      tabIndex={0}
      aria-label="Lecture notes"
      aria-orientation="vertical"
      aria-controls={article.id}
      aria-valuemin={Number(limits.min.toFixed(2))}
      aria-valuemax={Number(limits.max.toFixed(2))}
      aria-valuenow={Number(share.toFixed(2))}
      aria-valuetext={`${Math.round(share)}% lecture notes, ${100 - Math.round(share)}% workspace`}
      title="Drag to resize. Left/Right arrows adjust; Home/End reach the limits. Double-click to reset."
      onPointerDown={event => {
        if (event.button !== 0 || !event.isPrimary) return
        event.preventDefault()
        event.currentTarget.focus({ preventScroll: true })
        event.currentTarget.setPointerCapture(event.pointerId)
        drag.current = {
          pointer: event.pointerId,
          x: event.clientX,
          share,
          width: geometry.current.width,
        }
        container.dataset.resizing = 'true'
      }}
      onPointerMove={move}
      onPointerUp={finish}
      onPointerCancel={finish}
      onLostPointerCapture={finish}
      onDblClick={() => change(defaultShare)}
      onKeyDown={event => {
        const step = event.shiftKey ? 5 : 2
        const next =
          event.key === 'ArrowLeft'
            ? share - step
            : event.key === 'ArrowRight'
              ? share + step
              : event.key === 'Home'
                ? geometry.current.min
                : event.key === 'End'
                  ? geometry.current.max
                  : null
        if (next === null) return
        event.preventDefault()
        event.stopPropagation()
        change(next)
      }}
    >
      <span aria-hidden="true" />
    </div>
  )
}
