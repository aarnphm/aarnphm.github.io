let toggleAbort: AbortController | null = null

document.addEventListener('nav', ev => {
  toggleAbort?.abort()
  toggleAbort = null

  const button = document.querySelector<HTMLButtonElement>('#stacked-note-toggle')
  const container = document.getElementById('stacked-notes-container')
  const header = document.querySelector<HTMLElement>('.header')

  if (!button || !container || !header) return
  if (window.location.hostname.startsWith('notes.aarnphm.xyz')) return

  let pendingToggle = false
  const controller = new AbortController()
  toggleAbort = controller

  const switchCheckState = async () => {
    if (pendingToggle) return
    pendingToggle = true
    try {
      const isChecked = button.getAttribute('aria-checked') === 'true'
      const body = document.body
      const currentUrl = window.location.href

      if (!isChecked) {
        button.setAttribute('aria-checked', 'true')
        container.classList.add('active')
        body.classList.add('stack-mode')
        header.classList.add('grid', 'all-col')

        if (window.location.hash) {
          window.history.pushState('', document.title, currentUrl.split('#')[0])
        }
        await window.stacked.navigate(new URL(`/${ev.detail.url}`, window.location.toString()))
      } else {
        window.stacked.destroy()
        window.location.reload()
      }
    } finally {
      pendingToggle = false
    }
  }

  button.addEventListener('click', switchCheckState, { signal: controller.signal })
  // Micromorph can reuse this button for a different page control.
  window.addCleanup(() => controller.abort())
})
