function codeBlockClipboardSource(codeBlock: HTMLElement): string {
  return (codeBlock.dataset.clipboard ?? codeBlock.innerText).replace(/\n\n/g, '\n')
}

document.addEventListener('nav', () => {
  const els = document.getElementsByTagName('pre')
  for (let i = 0; i < els.length; i++) {
    const codeBlock = els[i].getElementsByTagName('code')[0]
    const button = els[i].querySelector('span.clipboard-button')
    if (codeBlock) {
      function onClick() {
        navigator.clipboard.writeText(codeBlockClipboardSource(codeBlock)).then(
          () => {
            button?.classList.add('check')
            setTimeout(() => {
              button?.classList.remove('check')
            }, 2000)
          },
          error => console.error(error),
        )
      }
      button?.addEventListener('click', onClick)
      window.addCleanup(() => button?.removeEventListener('click', onClick))
    }
  }
})
