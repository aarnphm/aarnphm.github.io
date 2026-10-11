export function mountTrailClock(clock: HTMLTimeElement, signal: AbortSignal) {
  const document = clock.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('The local clock needs a browser window')
  const format = new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h12',
  })
  const timezone = format.resolvedOptions().timeZone
  clock.dataset.timezone = timezone
  clock.title = `local time · ${timezone}`
  let timer: number | null = null
  const pause = () => {
    if (timer !== null) view.clearTimeout(timer)
    timer = null
  }
  const resume = () => {
    pause()
    if (signal.aborted || document.hidden) return
    const now = new Date()
    clock.textContent = format.format(now)
    clock.dateTime = now.toISOString()
    timer = view.setTimeout(resume, 1000 - (now.getTime() % 1000))
  }
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  signal.addEventListener('abort', pause, { once: true })
  resume()
}
