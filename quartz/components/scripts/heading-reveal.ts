/** Expands every collapsed section and folded transclude that hides `element`, so it can be measured and scrolled to. */
export function revealHeading(element: HTMLElement) {
  let content = element.closest('.collapsible-header-content')
  while (content) {
    const toggle = content
      .closest('.collapsible-header')
      ?.querySelector<HTMLElement>('.toggle-button')
    if (toggle?.getAttribute('aria-expanded') === 'false') toggle.click()
    content = content.parentElement?.closest('.collapsible-header-content') ?? null
  }

  element
    .closest('.transclude-collapsible.is-collapsed')
    ?.querySelector<HTMLElement>('.transclude-fold')
    ?.click()
}
