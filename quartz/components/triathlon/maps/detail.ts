export interface MapDetailTransition {
  cancel(): void
  close(animate?: boolean): void
}

export const createMapDetailTransition = (options: {
  detail: HTMLElement | null
  selection: HTMLElement | null
  reducedMotion: boolean
  onClosed: () => void
}): MapDetailTransition => {
  let animation: Animation | null = null
  const cancel = (): void => {
    animation?.cancel()
    animation = null
  }
  const close = (animate = false): void => {
    cancel()
    options.selection?.setAttribute('aria-hidden', 'true')
    const card = options.detail?.querySelector<HTMLElement>('.tri-pop-card')
    if (!animate || options.reducedMotion || !card) {
      options.onClosed()
      return
    }
    const current = card.animate(
      [
        { opacity: 1, transform: 'translateX(0)' },
        { opacity: 0, transform: 'translateX(1.25rem)' },
      ],
      { duration: 200, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' },
    )
    animation = current
    const finish = (): void => {
      if (animation !== current) return
      animation = null
      options.onClosed()
    }
    current.finished.then(finish).catch(finish)
  }
  return { cancel, close }
}
