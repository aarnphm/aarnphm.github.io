document.addEventListener('nav', () => {
  const buttons = document.querySelectorAll<HTMLButtonElement>(
    'ul.content-meta .readable-source button.clipboard-button',
  )
  for (const button of buttons) {
    const href = button.dataset.href
    const status = button
      .closest('.readable-source')
      ?.querySelector<HTMLElement>('.source-copy-status')
    if (!href || !status) continue
    const controller = new AbortController()
    let resetTimer: ReturnType<typeof setTimeout> | undefined
    let copying = false

    const onClick = async () => {
      if (copying) return
      clearTimeout(resetTimer)
      button.classList.remove('check')
      copying = true
      button.setAttribute('aria-busy', 'true')
      status.textContent = 'Copie…'
      try {
        if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
          throw new Error('Clipboard access is unavailable')
        }
        const source = fetch(href, { signal: controller.signal }).then(async response => {
          if (!response.ok) throw new Error(`Source request failed (${response.status})`)
          return new Blob([await response.text()], { type: 'text/plain' })
        })
        // Start the write during the click; the source can resolve after user activation expires.
        await Promise.all([
          source,
          navigator.clipboard.write([new ClipboardItem({ 'text/plain': source })]),
        ])
        if (controller.signal.aborted) return
        button.classList.add('check')
        status.textContent = 'Copié'
        resetTimer = setTimeout(() => {
          button.classList.remove('check')
          status.textContent = ''
        }, 2000)
      } catch (error) {
        if (controller.signal.aborted) return
        status.textContent = 'Copie impossible. Ouvrez llms.txt pour copier la source.'
        console.error('Could not copy Markdown source', error)
      } finally {
        copying = false
        button.removeAttribute('aria-busy')
      }
    }
    button.addEventListener('click', onClick)
    window.addCleanup(() => {
      button.removeEventListener('click', onClick)
      controller.abort()
      clearTimeout(resetTimer)
    })
  }
})
