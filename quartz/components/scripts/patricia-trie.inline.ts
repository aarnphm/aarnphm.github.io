import { registerMdxContent } from './mdx-content'

const ROOT_SELECTOR = '[data-patricia-trie]'

// Every tree state is rendered on the server; the script only shows the chosen one.
const setupPatriciaTrie = (root: HTMLElement): (() => void) => {
  const states = Array.from(root.querySelectorAll<HTMLElement>('[data-ptr-state]'))
  const apply = () => {
    const step = root.querySelector<HTMLInputElement>('[data-ptr-step-input]:checked')?.value
    if (step === undefined) return
    for (const s of states) s.hidden = s.getAttribute('data-ptr-state') !== step
  }
  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent(ROOT_SELECTOR, root => {
  window.addCleanup?.(setupPatriciaTrie(root))
})
