import { NOTES_HOSTNAME } from '../../util/stacked-notes'

let toggleAbort: AbortController | null = null

document.addEventListener('nav', () => {
  toggleAbort?.abort()
  toggleAbort = null

  const button = document.querySelector<HTMLButtonElement>('#stacked-note-toggle')
  const container = document.getElementById('stacked-notes-container')
  const header = document.querySelector<HTMLElement>('.header')

  if (!button || !container || !header) return
  if (window.location.hostname === NOTES_HOSTNAME) return

  let pendingToggle = false
  const controller = new AbortController()
  toggleAbort = controller

  const switchCheckState = async () => {
    if (pendingToggle) return
    pendingToggle = true
    try {
      const isChecked = button.getAttribute('aria-checked') === 'true'
      if (!isChecked) {
        await window.stacked.open()
      } else {
        window.stacked.destroy()
      }
    } finally {
      pendingToggle = false
    }
  }

  button.addEventListener('click', switchCheckState, { signal: controller.signal })
  // Micromorph can reuse this button for a different page control.
  window.addCleanup(() => controller.abort())
})
