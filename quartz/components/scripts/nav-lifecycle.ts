const lifecycle = window.quartzNavLifecycle ?? { listening: false }
window.quartzNavLifecycle = lifecycle

if (!lifecycle.listening) {
  lifecycle.listening = true
  document.addEventListener('nav', () => {
    if (lifecycle.controller && !lifecycle.controller.signal.aborted) return

    const controller = new AbortController()
    lifecycle.controller = controller
    window.addCleanup(() => {
      controller.abort()
      if (lifecycle.controller === controller) lifecycle.controller = undefined
    })
  })
}

export function currentNavSignal(): AbortSignal {
  if (!lifecycle.controller) throw new Error('Navigation lifecycle is not active')
  return lifecycle.controller.signal
}
