type GroundObject = { element: SVGGElement; owner: Element; index: number }

export function mountGroundDepth(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-depth-objects')
  if (!view || !layer) throw new Error('Ground depth needs the trail scene')
  let objects: GroundObject[] = []
  let frame: number | null = null
  const foot = (element: SVGGElement) => {
    const y = Number(element.dataset.groundY)
    if (Number.isFinite(y)) return y
    const transforms = element.transform.baseVal
    return transforms.numberOfItems ? transforms.getItem(0).matrix.f : 0
  }
  const sort = () => {
    frame = null
    if (host.hidden || document.hidden || signal.aborted) return
    const ordered = objects
      .filter(object => object.element.isConnected)
      .sort((a, b) => foot(a.element) - foot(b.element) || a.index - b.index)
    let next = layer.firstElementChild
    for (const { element } of ordered) {
      if (element === next) next = next.nextElementSibling
      else layer.insertBefore(element, next)
    }
    scene.dataset.depthOrderMotion = 'idle'
  }
  const schedule = () => {
    if (frame !== null || host.hidden || document.hidden || signal.aborted) return
    scene.dataset.depthOrderMotion = 'active'
    frame = view.requestAnimationFrame(sort)
  }
  const observer = new MutationObserver(schedule)
  const pause = () => {
    observer.disconnect()
    if (frame !== null) view.cancelAnimationFrame(frame)
    frame = null
    scene.dataset.depthOrderMotion = 'paused'
  }
  const restore = () => {
    pause()
    // Generation holders retain ownership while the paint layer orders individual feet.
    for (const { element, owner } of objects)
      if (element.isConnected && owner.isConnected) owner.append(element)
    objects = []
  }
  const refresh = () => {
    restore()
    for (const element of scene.querySelectorAll<SVGGElement>(
      '.trail-tree[data-tree-depth="meadow"], [data-trail-wander], .trail-map-reader, .trail-backpacker',
    )) {
      const owner = element.parentElement
      if (owner) objects.push({ element, owner, index: objects.length })
    }
    sort()
    if (!host.hidden && !document.hidden && !signal.aborted)
      observer.observe(scene, {
        subtree: true,
        attributes: true,
        attributeFilter: ['data-ground-y'],
      })
  }
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : refresh()), {
    signal,
  })
  signal.addEventListener('abort', pause, { once: true })
  return { refresh, restore, pause, resume: refresh }
}
