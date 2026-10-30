const status = document.querySelector('#status')
const enable = document.querySelector('#enable')
const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
let site
try {
  site = new URL(tab?.url)
} catch {}
if (!site || site.protocol !== 'https:') {
  enable.disabled = true
  status.textContent = 'Open an HTTPS page to add a site.'
} else {
  const result = await chrome.runtime.sendMessage({ type: 'bridge-status' })
  status.textContent =
    result.error ||
    (result.configuration.enabled ? 'Desktop app connected' : 'Open Audio Output to enable routing')
  enable.textContent = `Enable ${site.hostname}`
}
enable.addEventListener('click', async () => {
  try {
    const pattern = `${site.origin}/*`
    if (!(await chrome.permissions.request({ origins: [pattern] }))) return
    const id = `site-${site.hostname.replaceAll('.', '-')}`
    const registered = await chrome.scripting.getRegisteredContentScripts({ ids: [id] })
    if (!registered.length && site.hostname !== 'music.youtube.com') {
      await chrome.scripting.registerContentScripts([
        {
          id,
          matches: [pattern],
          js: ['content.js'],
          runAt: 'document_idle',
          persistAcrossSessions: true,
        },
      ])
    }
    const result = await chrome.runtime.sendMessage({ type: 'subscribe', host: site.hostname })
    if (!result.connected)
      throw new Error('Desktop bridge is unavailable. Run just install-bridge.')
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] })
    status.textContent = 'Site added. Choose its output in Audio Output.'
  } catch (error) {
    status.textContent = error.message
  }
})
