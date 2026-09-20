import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom'

export const mountWorkoutLapHover = (button: HTMLButtonElement, signal: AbortSignal): void => {
  const tooltip = button.querySelector<HTMLElement>('.tri-workout-lap-tooltip')
  const bar = button.querySelector<HTMLElement>('.tri-workout-lap-bar')
  if (!tooltip || !bar) return

  let pointer: { x: number; y: number } | null = null
  let cleanup: (() => void) | null = null
  let revision = 0
  const reference = {
    contextElement: button,
    getBoundingClientRect: (): DOMRect => {
      const rect = bar.getBoundingClientRect()
      return new DOMRect(pointer?.x ?? rect.left + rect.width / 2, pointer?.y ?? rect.top, 0, 0)
    },
  }
  const update = async (): Promise<void> => {
    const current = ++revision
    const { x, y } = await computePosition(reference, tooltip, {
      placement: 'right',
      middleware: [offset(8), flip({ crossAxis: false, padding: 2 }), shift({ padding: 2 })],
    })
    if (signal.aborted || current !== revision) return
    Object.assign(tooltip.style, { left: `${x}px`, top: `${y}px` })
  }
  const start = (): void => {
    if (cleanup) void update()
    else cleanup = autoUpdate(reference, tooltip, update)
  }
  const stop = (): void => {
    revision++
    cleanup?.()
    cleanup = null
  }
  const move = (event: PointerEvent): void => {
    if (event.pointerType === 'touch') return
    pointer = { x: event.clientX, y: event.clientY }
    start()
  }
  button.addEventListener('pointerenter', move, { signal })
  button.addEventListener('pointermove', move, { signal })
  button.addEventListener(
    'pointerleave',
    () => {
      pointer = null
      if (button.matches(':focus-visible')) start()
      else stop()
    },
    { signal },
  )
  button.addEventListener('focus', start, { signal })
  button.addEventListener(
    'blur',
    () => {
      if (!button.matches(':hover')) stop()
    },
    { signal },
  )
  signal.addEventListener('abort', stop, { once: true })
}
