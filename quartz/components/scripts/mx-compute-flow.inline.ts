import { registerMdxContent } from './mdx-content'

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

registerMdxContent('[data-mx-compute-flow]', root => {
  window.addCleanup?.(setupMXComputeFlow(root))
})
