export {}

// Every pass is rendered at build time; the root attribute selects which one the stylesheet shows.
const setupMXComputeFlow = (root: HTMLElement): (() => void) => {
  const apply = () => {
    const pass = root.querySelector<HTMLInputElement>('[data-mxf-pass-input]:checked')?.value
    if (pass) root.setAttribute('data-mxf-pass', pass)
  }
  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-mx-compute-flow]'))
    window.addCleanup?.(setupMXComputeFlow(root))
})
