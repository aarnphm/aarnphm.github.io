export function trailScene(document: Document) {
  const host = document.querySelector<HTMLElement>('.homepage-trail-scene')
  return (
    host?.shadowRoot?.querySelector<HTMLElement>('.trail-scene') ??
    host?.querySelector<HTMLElement>('.trail-scene') ??
    null
  )
}

export function trailHost(element: Element) {
  const root = element.getRootNode()
  const view = element.ownerDocument.defaultView
  if (view && root instanceof view.ShadowRoot && root.host instanceof view.HTMLElement)
    return root.host
  return element.closest<HTMLElement>('.homepage-trail-scene')
}

export function trailSceneHeight(host: HTMLElement, content: HTMLElement) {
  const view = host.ownerDocument.defaultView
  if (!view) return 0
  const bottom = content.getBoundingClientRect().bottom + view.scrollY
  return Math.max(view.innerHeight, bottom + 720, Number(host.dataset.trailWorldHeight) || 0)
}
