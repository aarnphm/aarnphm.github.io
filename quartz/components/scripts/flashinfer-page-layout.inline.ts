export {}

const ROOT = '[data-flashinfer-page-layout]'

const setup = (root: HTMLElement): (() => void) | null => {
  const setReq = (value: string) => {
    root.setAttribute('data-fpl-req', value)
    for (const page of root.querySelectorAll<SVGGElement>('[data-fpl-page]')) {
      const owner = page.getAttribute('data-fpl-owner') ?? ''
      const dim = value !== '' && owner !== value
      page.setAttribute('data-fpl-dim', dim ? 'true' : 'false')
    }
    for (const cell of root.querySelectorAll<HTMLElement>('[data-fpl-qo]')) {
      const owner = cell.getAttribute('data-fpl-owner') ?? ''
      const dim = value !== '' && owner !== value
      cell.setAttribute('data-fpl-dim', dim ? 'true' : 'false')
    }
    for (const row of root.querySelectorAll<HTMLElement>('[data-fpl-row]')) {
      const id = row.getAttribute('data-fpl-row') ?? ''
      row.toggleAttribute('data-fpl-on', value === '' || value === id)
    }
  }

  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement | null
    if (!input?.matches('[data-fpl-req-input]')) return
    setReq(input.value)
  }

  root.addEventListener('change', onChange)
  setReq(root.getAttribute('data-fpl-req') ?? '')
  return () => root.removeEventListener('change', onChange)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(ROOT)) {
    const cleanup = setup(root)
    if (cleanup) window.addCleanup(cleanup)
  }
})
