/** Bind a synchronous source to native button activation and page-scoped copy feedback. */
export function bindClipboardButton(button: HTMLButtonElement, source: () => string): () => void {
  let resetTimer: ReturnType<typeof setTimeout> | undefined
  let revision = 0
  let disposed = false

  const onClick = async () => {
    const request = ++revision
    clearTimeout(resetTimer)
    button.classList.remove('check')
    try {
      await navigator.clipboard.writeText(source())
      if (disposed || request !== revision) return
      button.classList.add('check')
      resetTimer = setTimeout(() => button.classList.remove('check'), 2000)
    } catch (error) {
      console.error('Could not copy source', error)
    }
  }

  button.addEventListener('click', onClick)
  return () => {
    disposed = true
    clearTimeout(resetTimer)
    button.classList.remove('check')
    button.removeEventListener('click', onClick)
  }
}
