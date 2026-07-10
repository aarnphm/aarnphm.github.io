const BADGE_SELECTOR = '.alphaxiv-link-badge, [data-alphaxiv-link-badge]'

function removeBadges(root: ParentNode) {
  root.querySelectorAll(BADGE_SELECTOR).forEach(badge => badge.remove())
}

const observer = new MutationObserver(mutations => {
  for (const mutation of mutations) {
    mutation.addedNodes.forEach(node => {
      if (!(node instanceof Element)) return
      if (node.matches(BADGE_SELECTOR)) node.remove()
      else removeBadges(node)
    })
  }
})

removeBadges(document)
observer.observe(document.body, { childList: true, subtree: true })
document.addEventListener('nav', () => removeBadges(document))
