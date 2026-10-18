import { loadPageStyles, pageStylesPending } from './page-styles'

// Popovers, previews and stacked notes insert markup from other pages, which can need a
// page-scoped sheet this page did not link.
document.addEventListener('nav', () => {
  void loadPageStyles(document.body)
  if (!pageStylesPending()) return
  const observer = new MutationObserver(records => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node instanceof Element && (node.hasAttribute('class') || node.childElementCount > 0)) {
          void loadPageStyles(node)
        }
      }
    }
    if (!pageStylesPending()) observer.disconnect()
  })
  observer.observe(document.body, { childList: true, subtree: true })
  window.addCleanup(() => observer.disconnect())
})
