type GroundObject = { element: SVGGElement; owner: Element; layer: SVGGElement; index: number }

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
    for (const paint of new Set(objects.map(object => object.layer))) {
      const ordered = objects
        .filter(object => object.layer === paint && object.element.isConnected)
        .sort((a, b) => foot(a.element) - foot(b.element) || a.index - b.index)
      let next = paint.firstElementChild
      for (const { element } of ordered) {
        if (element === next) next = next.nextElementSibling
        else if (typeof paint.moveBefore === 'function') paint.moveBefore(element, next)
        else paint.insertBefore(element, next)
      }
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
      if (element.isConnected && owner.isConnected) {
        if (typeof owner.moveBefore === 'function') owner.moveBefore(element, null)
        else owner.append(element)
      }
    objects = []
  }
  const refresh = () => {
    restore()
    const distant = scene.querySelector<SVGGElement>('.trail-foothill-objects')
    for (const element of scene.querySelectorAll<SVGGElement>(
      '.trail-tree, [data-trail-wander], [data-holiday-decoration], .trail-map-reader, .trail-backpacker, [data-alpine-traveler], [data-camp-person], [data-camp-tent], .trail-extra-camp, [data-park-ranger], [data-trail-visitor]',
    )) {
      const owner = element.parentElement
      const far =
        element.dataset.treeDepth === 'foothill' || element.hasAttribute('data-alpine-traveler')
      if (owner && (!far || distant))
        objects.push({
          element,
          owner,
          layer: far && distant ? distant : layer,
          index: objects.length,
        })
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
