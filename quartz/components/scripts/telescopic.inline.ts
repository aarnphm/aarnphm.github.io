document.addEventListener('nav', () => {
  const telescopics = document.querySelectorAll<HTMLElement>('.telescopic-container')
  if (telescopics.length === 0) return

  telescopics.forEach(tel => {
    const el = tel.querySelector<HTMLElement>('div#telescope')
    if (!el) return

    const expandable = el.querySelectorAll<HTMLSpanElement>('span[class~="details"]')

    expandable.forEach(closed => {
      function onClick() {
        closed.classList.remove('close')
        closed.classList.add('open')
      }
      closed.addEventListener('click', onClick)
      window.addCleanup(() => closed.removeEventListener('click', onClick))
    })
  })
})
