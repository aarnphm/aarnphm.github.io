document.addEventListener('nav', async () => {
  const els = document.querySelectorAll('ul.content-meta')
  if (!els) return

  for (let i = 0; i < els.length; i++) {
    const button = els[i].querySelector('span.clipboard-button') as HTMLSpanElement
    if (!button) continue

    async function onClick() {
      const href = button.dataset.href as string
      const res = await fetch(href)
      const text = await res.text()
      navigator.clipboard.writeText(text).then(
        () => {
          button?.classList.add('check')
          setTimeout(() => {
            button?.classList.remove('check')
          }, 2000)
        },
        error => console.error(error),
      )
    }
    button.addEventListener('click', onClick)
    window.addCleanup(() => button.removeEventListener('click', onClick))
  }
})
