const BADGE_SELECTOR = '.alphaxiv-link-badge, [data-alphaxiv-link-badge]'

function isBadge(el: Element): boolean {
  return el.classList.contains('alphaxiv-link-badge') || el.hasAttribute('data-alphaxiv-link-badge')
}

function removeBadges(root: ParentNode) {
  root.querySelectorAll(BADGE_SELECTOR).forEach(badge => badge.remove())
}

const observer = new MutationObserver(mutations => {
  for (const mutation of mutations) {
    for (const node of mutation.addedNodes) {
      if (!(node instanceof Element)) continue
      if (isBadge(node)) node.remove()
      else if (node.childElementCount > 0) removeBadges(node)
    }
  }
})

removeBadges(document)
observer.observe(document.body, { childList: true, subtree: true })
document.addEventListener('nav', () => removeBadges(document))
