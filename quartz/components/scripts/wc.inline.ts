document.addEventListener('nav', () => {
  const modal = document.getElementById('wc-modal') as HTMLElement
  if (!modal) return
  const inner = modal.querySelector('.wc-inner') as HTMLElement

  let isVisible = false
  function hideModal() {
    if (!isVisible) return
    isVisible = false
    modal!.style.visibility = 'hidden'
  }

  function updateModal() {
    const selection = window.getSelection()

    if (!selection || selection.isCollapsed) {
      hideModal()
      return
    }
    const anchorEl =
      selection.anchorNode instanceof Element
        ? selection.anchorNode
        : selection.anchorNode?.parentElement
    if (anchorEl?.closest('.cm-editor, textarea, input')) {
      hideModal()
      return
    }
    const text = selection.toString().trim()
    if (!text) {
      hideModal()
      return
    }
    inner.textContent = `${text.split(' ').filter(word => word.length > 0).length} words`
    if (!isVisible) {
      isVisible = true
      modal!.style.visibility = 'visible'
    }
  }

  document.addEventListener('selectionchange', updateModal)
  window.addCleanup(() => {
    document.removeEventListener('selectionchange', updateModal)
  })
})
