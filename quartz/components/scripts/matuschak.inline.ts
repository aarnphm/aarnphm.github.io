let toggleAbort: AbortController | null = null
let pendingToggle = false

document.addEventListener('nav', async ev => {
  const button = document.getElementById('stacked-note-toggle') as HTMLButtonElement
  const container = document.getElementById('stacked-notes-container')
  const header = document.getElementsByClassName('header')[0] as HTMLElement

  if (!button || !container || !header) return

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

  if (window.location.hostname.startsWith('notes.aarnphm.xyz')) return

  toggleAbort?.abort()
  toggleAbort = new AbortController()
  button.addEventListener('click', switchCheckState, { signal: toggleAbort.signal })
})
