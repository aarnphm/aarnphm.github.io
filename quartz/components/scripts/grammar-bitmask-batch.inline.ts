export {}

const setupGrammarBitmaskBatch = (root: HTMLElement): (() => void) => {
  const requestParts = Array.from(root.querySelectorAll<Element>('[data-gbb-req]'))
  const workerParts = Array.from(root.querySelectorAll<Element>('[data-gbb-n]'))
  const checked = (kind: string) =>
    root.querySelector<HTMLInputElement>(`[data-gbb-input="${kind}"]:checked`)?.value

  const apply = () => {
    const request = checked('request')
    const workers = checked('workers')
    if (!request || !workers) return
    root.setAttribute('data-gbb-request', request)
    root.setAttribute('data-gbb-workers', workers)
    for (const el of requestParts)
      el.setAttribute('data-on', el.getAttribute('data-gbb-req') === request ? 'true' : 'false')
    for (const el of workerParts)
      el.setAttribute('data-on', el.getAttribute('data-gbb-n') === workers ? 'true' : 'false')
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-grammar-bitmask-batch]')) {
    window.addCleanup?.(setupGrammarBitmaskBatch(root))
  }
})
