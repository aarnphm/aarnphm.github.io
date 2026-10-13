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
