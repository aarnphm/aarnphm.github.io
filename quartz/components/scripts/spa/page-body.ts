import micromorph from 'micromorph'

export function replacePageBody(nextBody: HTMLElement): void {
  const page = document.body.querySelector<HTMLElement>(':scope > #quartz-root')
  const nextPage = nextBody.querySelector<HTMLElement>(':scope > #quartz-root')
  // Slides and authored pages can carry executable inline scripts inside their content.
  const hasPageScripts = nextPage?.querySelector(
    'script:not([type="application/json"]):not([type="application/ld+json"])',
  )
  if (!page || !nextPage || hasPageScripts) {
    void micromorph(document.body, nextBody)
    return
  }

  // Page components have been cleaned up. Replace their markup in bulk and leave the body
  // attributes, persistent scripts and standalone-viewer behavior with the existing morph.
  const content = Array.from(nextPage.childNodes)
  nextPage.replaceChildren()
  page.replaceChildren()
  void micromorph(document.body, nextBody)
  const mountedPage = document.getElementById('quartz-root')
  if (!mountedPage) throw new Error('Page root missing after navigation')
  mountedPage.replaceChildren(...content)
}
