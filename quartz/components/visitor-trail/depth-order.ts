// Landforms (`.trail-landform`) sort by the last hidden row behind their crest; see relief.ts.
export const groundObjectSelector =
  '.trail-landform, .trail-tree, .trail-understory, [data-trail-wander], [data-wetland-animal], .trail-river-bear, .trail-river-fish, [data-holiday-decoration], .trail-map-reader, .trail-backpacker, [data-alpine-traveler], [data-camp-person], [data-camp-tent], .trail-extra-camp, [data-park-ranger], [data-trail-visitor], [data-trail-pet], [data-trail-athlete]'

type GroundObject = {
  element: SVGGElement
  owner: Element
  layer: SVGGElement
  index: number
  y: number
}

export function mountGroundDepth(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const layer = scene.querySelector<SVGGElement>('.trail-depth-objects')
  if (!layer) throw new Error('Ground depth needs the trail scene')
  let objects: GroundObject[] = []
  let byElement = new WeakMap<Element, GroundObject>()
  const byLayer = new Map<SVGGElement, GroundObject[]>()
  const foot = (element: SVGGElement) => {
    const y = Number(element.dataset.groundY)
    if (Number.isFinite(y)) return y
    const transforms = element.transform.baseVal
    return transforms.numberOfItems ? transforms.getItem(0).matrix.f : 0
  }
  const compare = (a: GroundObject, b: GroundObject) => a.y - b.y || a.index - b.index
  const applyLayer = (paint: SVGGElement, ordered: GroundObject[]) => {
    ordered.sort(compare)
    if (scene.dataset.trailOpening !== 'complete' && scene.dataset.trailRasterReady !== 'true') {
      paint.prepend(...ordered.map(({ element }) => element).filter(element => element.isConnected))
      return
    }
    let next = paint.firstElementChild
    for (const { element } of ordered) {
      if (!element.isConnected) continue
      if (element === next) next = next.nextElementSibling
      else if (typeof paint.moveBefore === 'function') paint.moveBefore(element, next)
      else paint.insertBefore(element, next)
    }
  }
  const sort = (records?: MutationRecord[]) => {
    if (document.hidden || signal.aborted) return
    if (records) {
      const touched = new Set<SVGGElement>()
      for (const record of records) {
        const object = byElement.get(record.target as Element)
        if (!object || !object.element.isConnected) continue
        const nextY = foot(object.element)
        if (nextY === object.y) continue
        object.y = nextY
        touched.add(object.layer)
      }
      for (const paint of touched) {
        const ordered = byLayer.get(paint)
        if (!ordered) continue
        let outOfOrder = false
        for (let index = 1; index < ordered.length; index++) {
          if (compare(ordered[index - 1]!, ordered[index]!) > 0) {
            outOfOrder = true
            break
          }
        }
        if (outOfOrder) applyLayer(paint, ordered)
      }
      scene.dataset.depthOrderMotion = 'idle'
      return
    }
    byLayer.clear()
    for (const object of objects) {
      if (!object.element.isConnected) continue
      object.y = foot(object.element)
      const list = byLayer.get(object.layer)
      if (list) list.push(object)
      else byLayer.set(object.layer, [object])
    }
    for (const [paint, ordered] of byLayer) applyLayer(paint, ordered)
    scene.dataset.depthOrderMotion = 'idle'
  }
  // Apply new feet before the SVG paints; another animation frame leaves one stale depth order.
  const observer = new MutationObserver(records => sort(records))
  const pause = () => {
    observer.disconnect()
    scene.dataset.depthOrderMotion = 'paused'
  }
  const restore = (selector?: string) => {
    pause()
    // Generation holders retain ownership while the paint layer orders individual feet.
    for (const { element, owner } of objects)
      if (element.isConnected && owner.isConnected && (!selector || element.matches(selector))) {
        if (typeof owner.moveBefore === 'function') owner.moveBefore(element, null)
        else owner.append(element)
      }
    objects = selector
      ? objects.filter(({ element }) => element.isConnected && !element.matches(selector))
      : []
    byElement = new WeakMap(objects.map(object => [object.element, object]))
    byLayer.clear()
  }
  const refresh = () => {
    pause()
    objects = objects.filter(({ element, owner }) => element.isConnected && owner.isConnected)
    byElement = new WeakMap(objects.map(object => [object.element, object]))
    const distant = scene.querySelector<SVGGElement>('.trail-foothill-objects')
    const paintLayer = (element: SVGGElement) =>
      distant &&
      (element.dataset.treeDepth === 'foothill' ||
        element.hasAttribute('data-alpine-traveler') ||
        element.dataset.petAlpine === 'true')
        ? distant
        : layer
    for (const object of objects) object.layer = paintLayer(object.element)
    for (const element of scene.querySelectorAll<SVGGElement>(groundObjectSelector)) {
      if (byElement.has(element)) continue
      const owner = element.parentElement
      if (owner) {
        const object: GroundObject = {
          element,
          owner,
          layer: paintLayer(element),
          index: objects.length,
          y: foot(element),
        }
        objects.push(object)
        byElement.set(element, object)
      }
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
  scene.addEventListener('trail:depth-change', refresh, { signal })
  signal.addEventListener('abort', pause, { once: true })
  return { refresh, restore, pause, resume: refresh }
}
