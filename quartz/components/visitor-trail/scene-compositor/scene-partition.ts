import type { Gpu } from './gpu-textures'
import { groundObjectSelector } from '../depth-order'
import { ctm, identity, invert, multiply, translation, type Matrix } from './affine-geometry'
import {
  reference,
  unpainted,
  type Box,
  type Group,
  type Layer,
  type Run,
  type Scope,
  type Surface,
} from './render-model'

// The texture tree: which elements keep their own textures, the runs their siblings share,
// and what a mutation or an animation dirties.
export function createSceneTree(scope: Scope, gpu: Gpu) {
  const { view, svg, is, tree, shared } = scope
  const { release } = gpu
  // Elements whose own transform, opacity or position changes keep their own textures. A fade
  // keeps its element apart only while it runs, so the opening does not leave the scene in shards.
  const promoted = new Map<Element, 'shape' | 'motion' | 'hot'>()
  const fading = new WeakMap<Element, number>()
  let carriers = new WeakSet<Element>()
  const groups = new Map<Element, Group>()
  const runOf = new WeakMap<Element, Run>()
  const carry = (element: Element) => {
    for (
      let parent: Element | null = element.parentElement;
      parent && parent !== svg;
      parent = parent.parentElement
    )
      carriers.add(parent)
  }
  const promote = (element: Element, reason: 'shape' | 'motion' | 'hot' = 'shape') => {
    if (reason === 'motion') fading.set(element, shared.clock)
    const known = promoted.get(element)
    if (known) {
      if (known === 'motion' && reason !== 'motion') promoted.set(element, reason)
      return
    }
    if (element === svg || !svg.contains(element)) return
    if (element.closest('defs, mask, clipPath')) return
    promoted.set(element, reason)
    carry(element)
    let owner = element.parentElement
    while (owner && !groups.has(owner)) owner = owner.parentElement
    const group = owner && groups.get(owner)
    if (group) group.structure = true
  }
  const discover = (root: Element) => {
    // Depth sorting moves known objects every frame; only new markup needs a search.
    if (!is(root, 'SVGElement')) return
    if (promoted.has(root)) {
      carry(root)
      let owner = root.parentElement
      while (owner && !groups.has(owner)) owner = owner.parentElement
      const group = owner && groups.get(owner)
      if (group) group.structure = true
      return
    }
    if (
      root.matches(`${groundObjectSelector}, .trail-tree-canopy`) ||
      root.hasAttribute('data-trail-depth') ||
      root.hasAttribute('data-trail-chunk') ||
      root.hasAttribute('data-trail-layer')
    )
      promote(root)
    if (root.hasAttribute('mask') || is(root, 'SVGForeignObjectElement')) promote(root)
    for (const element of root.querySelectorAll(
      `${groundObjectSelector}, .trail-tree-canopy, [data-trail-depth], [data-trail-chunk], [data-trail-layer], [mask], foreignObject`,
    ))
      promote(element)
  }

  // A node's geometry box in the user space that `toSpace` maps the viewport into.
  let bboxes = new WeakMap<Element, DOMRect | null>()
  const resetBoxes = () => {
    bboxes = new WeakMap()
  }
  const invalidateBox = (element: Element) => {
    for (let node: Element | null = element; node && node !== svg; node = node.parentElement)
      bboxes.delete(node)
  }
  const localBox = (node: SVGGraphicsElement): DOMRect | null => {
    const cached = bboxes.get(node)
    if (cached !== undefined) return cached
    let box: DOMRect | null = null
    try {
      box = node.getBBox()
    } catch {
      box = null
    }
    if (box && (box.width || box.height)) bboxes.set(node, box)
    return box
  }
  const boxOf = (node: Element, toSpace: Matrix, self = false): Box | null => {
    if (!is(node, 'SVGGraphicsElement')) return null
    const box = localBox(node)
    if (!box || (!box.width && !box.height)) return null
    const matrix = self ? identity : multiply(toSpace, ctm(node))
    let left = Infinity
    let top = Infinity
    let right = -Infinity
    let bottom = -Infinity
    for (const [x, y] of [
      [box.x, box.y],
      [box.x + box.width, box.y],
      [box.x, box.y + box.height],
      [box.x + box.width, box.y + box.height],
    ]) {
      const px = matrix[0] * x + matrix[2] * y + matrix[4]
      const py = matrix[1] * x + matrix[3] * y + matrix[5]
      left = Math.min(left, px)
      top = Math.min(top, py)
      right = Math.max(right, px)
      bottom = Math.max(bottom, py)
    }
    return { left, top, right, bottom }
  }
  const group = (element: SVGGraphicsElement): Group => {
    const known = groups.get(element)
    if (known) return known
    const next: Group = {
      kind: 'group',
      element,
      items: [],
      structure: true,
      style: view.getComputedStyle(element),
      mask: null,
      clip: null,
      moved: true,
      fresh: false,
      matrix: identity,
      screen: identity,
      opacity: 1,
      shown: true,
    }
    groups.set(element, next)
    return next
  }
  const layer = (): Layer => ({
    texture: null,
    width: 0,
    height: 0,
    base: identity,
    scale: 0,
    palette: -1,
    revision: -1,
    dirty: true,
    waiting: 0,
  })
  const drop = (item: Run | Surface | Group) => {
    if (item.kind === 'group') {
      // A retired container can still reference descendants moved into another layer.
      if (
        svg.contains(item.element) &&
        (promoted.has(item.element) || carriers.has(item.element))
      ) {
        item.moved = true
        item.fresh = false
        return
      }
      for (const child of item.items) drop(child)
      release(item.mask?.texture ?? null)
      groups.delete(item.element)
    } else release(item.texture)
  }
  const partition = (owner: Group) => {
    owner.structure = false
    const previous = owner.items
    const reuse = new Map<Element, Run>()
    for (const item of previous) if (item.kind === 'run') reuse.set(item.nodes[0], item)
    const kept = new Set<Run | Surface | Group>()
    const items: (Run | Surface | Group)[] = []
    let pending: Element[] = []
    let extent: Box | null = null
    let covered = 0
    const flush = () => {
      const box = extent
      extent = null
      covered = 0
      if (!pending.length) return
      const known = reuse.get(pending[0])
      const same =
        known &&
        known.nodes.length === pending.length &&
        known.nodes.every((node, index) => node === pending[index])
      const run: Run = same
        ? known
        : {
            kind: 'run',
            owner,
            nodes: pending,
            box,
            hits: [],
            hot: new Set(),
            placed: null,
            ...layer(),
          }
      if (same && box) run.box = box
      for (const node of pending) runOf.set(node, run)
      items.push(run)
      kept.add(run)
      pending = []
    }
    const element = owner.element
    const leaf = !(
      is(element, 'SVGGElement') ||
      is(element, 'SVGSVGElement') ||
      is(element, 'SVGAElement')
    )
    const children = leaf ? [element] : [...element.children]
    const space = leaf ? identity : invert(ctm(element))
    for (const child of children) {
      if (
        child !== element &&
        (unpainted.has(child.localName) || child.hasAttribute('data-trail-model'))
      )
        continue
      if (child !== element && is(child, 'SVGForeignObjectElement')) {
        flush()
        const surface = child.querySelector('canvas')
        if (!surface) continue
        const known = previous.find(
          (item): item is Surface => item.kind === 'surface' && item.frame === child,
        )
        const shift = view.getComputedStyle(surface).transform
        const offset = new view.DOMMatrixReadOnly(shift === 'none' ? undefined : shift)
        const item: Surface = known ?? {
          kind: 'surface',
          frame: child,
          canvas: surface,
          style: view.getComputedStyle(child),
          offset: translation(child.x.baseVal.value + offset.e, child.y.baseVal.value + offset.f),
          shown: true,
          placed: null,
          texture: null,
          dirty: true,
          damage: null,
        }
        items.push(item)
        kept.add(item)
        continue
      }
      if (child !== element && (promoted.has(child) || carriers.has(child))) {
        flush()
        const item = group(child as SVGGraphicsElement)
        items.push(item)
        kept.add(item)
        continue
      }
      // Siblings share a texture while they overlap; far-apart ones would leave it mostly empty.
      const box = leaf ? null : boxOf(child, space)
      if (box) {
        const area = (box.right - box.left) * (box.bottom - box.top)
        const union: Box = extent
          ? {
              left: Math.min(extent.left, box.left),
              top: Math.min(extent.top, box.top),
              right: Math.max(extent.right, box.right),
              bottom: Math.max(extent.bottom, box.bottom),
            }
          : box
        if (
          extent &&
          (union.right - union.left) * (union.bottom - union.top) > 2 * (covered + area) + 4096
        )
          flush()
        extent = extent ? union : box
        covered += area
      }
      pending.push(child)
    }
    flush()
    for (const item of previous) {
      if (kept.has(item)) continue
      drop(item)
    }
    owner.items = items
    const clipId = element === svg ? undefined : reference(owner.style.clipPath)
    const clip = clipId ? tree().getElementById(clipId) : null
    owner.clip = is(clip, 'SVGClipPathElement') ? clip : null
    owner.fresh = false
    const maskId =
      element === svg ? undefined : reference(owner.style.mask === 'none' ? null : owner.style.mask)
    const source = maskId ? tree().getElementById(maskId) : null
    if (is(source, 'SVGMaskElement')) {
      if (owner.mask?.source !== source) {
        release(owner.mask?.texture ?? null)
        owner.mask = { ...layer(), source, placed: null }
      }
    } else if (owner.mask) {
      release(owner.mask.texture)
      owner.mask = null
    }
  }
  const root = group(svg)

  // The run that paints an element: its ancestor whose parent keeps its own textures.
  const ownerOf = (element: Element) => {
    for (let node: Element | null = element; node && node !== svg; node = node.parentElement) {
      const parent = node.parentElement
      if (parent && groups.has(parent)) return { run: runOf.get(node), top: node }
    }
    return { run: undefined, top: null }
  }
  const touch = (element: Element) => {
    invalidateBox(element)
    const group = groups.get(element)
    if (group) {
      dirtyGroup(group)
      return
    }
    const { run, top } = ownerOf(element)
    if (!run || !top) return
    run.dirty = true
    run.box = null
    run.hot.add(top)
    if (run.hits.at(-1) !== shared.clock) run.hits.push(shared.clock)
  }
  const dirtyGroup = (group: Group) => {
    invalidateBox(group.element)
    group.moved = true
    group.fresh = false
    if (group.mask) group.mask.dirty = true
    for (const item of group.items) {
      if (item.kind === 'group') dirtyGroup(item)
      else {
        item.dirty = true
        if (item.kind === 'run') item.box = null
      }
    }
  }
  const dirtyMask = (mask: Element) => {
    for (const group of groups.values()) if (group.mask?.source === mask) group.mask.dirty = true
  }
  const clipVersions = new WeakMap<SVGClipPathElement, number>()
  const touchDefinition = (element: Element) => {
    const mask = element.closest('mask')
    if (mask) return dirtyMask(mask)
    const clip = element.closest('clipPath')
    if (clip) {
      clipVersions.set(clip, (clipVersions.get(clip) ?? 0) + 1)
      return
    }
    if (element.closest('linearGradient, radialGradient')) return dirtyGroup(root)
    const id = element.closest('[id]')?.id
    if (id)
      for (const path of svg.querySelectorAll(`textPath[href="#${CSS.escape(id)}"]`)) touch(path)
  }
  const repaint = () => {
    dirtyGroup(root)
    shared.flush = true
    shared.revision += 1
  }
  const split = () => {
    for (const group of groups.values())
      for (const item of group.items) {
        if (item.kind !== 'run' || !item.hits.length) continue
        item.hits = item.hits.filter(time => shared.clock - time < 3000)
        if (item.hits.length < 3) {
          if (!item.hits.length) item.hot.clear()
          continue
        }
        // A node that changes again and again gets its own texture, so its run stops redrawing.
        const single = item.nodes.length === 1 && item.nodes[0] === group.element
        if (!single) for (const node of item.hot) promote(node, 'hot')
        item.hits = []
        item.hot.clear()
      }
  }
  // Returns finished fades and removed elements to their neighbours' textures.
  const settle = () => {
    let changed = false
    for (const [element, reason] of promoted) {
      const idle = reason === 'motion' && shared.clock - (fading.get(element) ?? 0) > 2000
      if (!idle && element.isConnected) continue
      promoted.delete(element)
      changed = true
    }
    if (!changed) return
    carriers = new WeakSet()
    for (const element of promoted.keys()) carry(element)
    // Only parents losing a texture boundary need new runs. Repartitioning every group
    // here measures the whole forest again whenever replaced crowns are collected.
    for (const group of groups.values())
      if (
        group.items.some(
          item =>
            item.kind === 'group' &&
            (!svg.contains(item.element) ||
              (!promoted.has(item.element) && !carriers.has(item.element))),
        )
      )
        group.structure = true
  }
  return {
    promoted,
    fading,
    groups,
    root,
    clipVersions,
    promote,
    discover,
    resetBoxes,
    invalidateBox,
    localBox,
    boxOf,
    partition,
    touch,
    dirtyGroup,
    touchDefinition,
    repaint,
    split,
    settle,
  }
}
export type SceneTree = ReturnType<typeof createSceneTree>
