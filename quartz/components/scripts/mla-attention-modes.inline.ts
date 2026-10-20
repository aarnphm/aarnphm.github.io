export {}

const MAM_ROOT_SELECTOR = '[data-mla-attention-modes]'

// Both panels are server-rendered; the radio group only switches the root's mode attribute.
const setupMLAAttentionModes = (root: HTMLElement): (() => void) => {
  const onChange = (event: Event) => {
    const target = event.target as HTMLInputElement | null
    if (target?.matches('[data-mam-mode-input]') && target.checked)
      root.setAttribute('data-mam-mode', target.value)
  }
  root.addEventListener('change', onChange)
  const checked = root.querySelector<HTMLInputElement>('[data-mam-mode-input]:checked')
  if (checked) root.setAttribute('data-mam-mode', checked.value)
  return () => root.removeEventListener('change', onChange)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(MAM_ROOT_SELECTOR)) {
    window.addCleanup?.(setupMLAAttentionModes(root))
  }
})
