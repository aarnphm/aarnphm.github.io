import type { SceneTree } from './scene-partition'
import type { Transitions } from './transitions'
import { motionProperties, motionVariables, type Scope } from './render-model'

// Mutation records: which attribute, style and child changes can change pixels, and the textures
// they dirty.
export function createMutations(
  scope: Scope,
  sceneTree: SceneTree,
  transitions: Transitions,
  schedule: () => void,
) {
  const { scene, document, svg, is, tree, shared } = scope
  const {
    groups,
    root,
    promote,
    discover,
    resetBoxes,
    invalidateBox,
    touch,
    touchDefinition,
    repaint,
  } = sceneTree
  const { preparePresentation } = transitions
  // Attributes that no selector reads cannot change a pixel. Journeys write dozens of them each frame.
  // The scene stylesheets load before the first render, so selectors are read then.
  let attributeSelectors: Map<string, string[]> | null = null
  const collect = (rules: CSSRuleList, into: Map<string, string[]>) => {
    for (const rule of rules) {
      if ('selectorText' in rule && typeof rule.selectorText === 'string')
        for (const name of new Set(
          Array.from(rule.selectorText.matchAll(/\[([\w-]+)/g), m => m[1]),
        ))
          into.set(name, [...(into.get(name) ?? []), rule.selectorText])
      if ('cssRules' in rule && rule.cssRules) collect(rule.cssRules as CSSRuleList, into)
    }
  }
  const readSelectors = () => {
    if (attributeSelectors) return
    const found = new Map<string, string[]>()
    for (const sheet of tree().styleSheets) {
      try {
        collect(sheet.cssRules, found)
      } catch {
        // A cross-origin sheet cannot be read; its selectors do not target the trail.
      }
    }
    found.delete('data-trail-renderer')
    found.delete('data-trail-opening')
    found.delete('data-trail-rendered')
    found.delete('data-trail-visible')
    attributeSelectors = found
  }
  const inert = (name: string) =>
    attributeSelectors !== null && name.startsWith('data-') && !attributeSelectors.has(name)
  const scratchStyle = document.createElement('div').style
  const motionOnly = (previous: string | null, next: string | null) => {
    const read = (value: string | null) => {
      scratchStyle.cssText = value ?? ''
      return new Map(
        [...scratchStyle]
          .filter(name => !motionProperties.has(name) && !motionVariables.has(name))
          .map(name => [name, scratchStyle.getPropertyValue(name)]),
      )
    }
    const before = read(previous)
    const after = read(next)
    return (
      before.size === after.size && [...before].every(([name, value]) => after.get(name) === value)
    )
  }
  // An attribute outside the SVG, such as a hovered blaze on the scene, restyles only the elements
  // its rules select. Without the attribute test, each selector matches them before and after the
  // change. Returns false when the whole scene may change.
  const restyle = (name: string) => {
    const selectors = attributeSelectors?.get(name)
    if (!selectors || !name.startsWith('data-')) return false
    const test = new RegExp(
      `:not\\(\\[${name}(?![\\w-])[^\\]]*\\]\\)|\\[${name}(?![\\w-])[^\\]]*\\]`,
      'g',
    )
    const touched = new Set<Element>()
    for (const selector of selectors) {
      let matches: NodeListOf<Element>
      try {
        matches = tree().querySelectorAll(selector.replace(test, ':is(*)'))
      } catch {
        return false
      }
      for (const element of matches) {
        if (element.contains(svg)) return false
        if (svg.contains(element)) touched.add(element)
      }
    }
    for (const element of touched) touch(element)
    return true
  }
  const record = (records: MutationRecord[]) => {
    if (scene.dataset.trailRendered !== 'true') {
      shared.generationPending = true
      return
    }
    if (shared.generationPending) {
      shared.generationPending = false
      shared.clipVersion += 1
      resetBoxes()
      discover(svg)
      for (const owner of groups.values()) {
        owner.structure = true
        owner.moved = true
        owner.fresh = false
      }
      repaint()
    }
    let changed = false
    for (const entry of records) {
      const target = is(entry.target, 'Element') ? entry.target : entry.target.parentElement
      if (!target) continue
      if (entry.type === 'attributes') {
        const name = entry.attributeName ?? ''
        const paletteChange = target === document.documentElement
        // A holiday or a chosen season restyles the whole scene: crossfade once it is painted.
        const holidayChange =
          target === scene && (name === 'data-holiday' || name === 'data-foliage-season')
        if (
          target.getAttribute(name) === entry.oldValue ||
          (!paletteChange && !holidayChange && inert(name))
        )
          continue
        changed = true
        if (paletteChange) {
          preparePresentation()
          shared.palettePending = true
          shared.paletteTransition = true
        } else if (holidayChange) {
          shared.paletteTransition ||= scene.dataset.trailOpening === 'complete'
          if (!restyle(name)) repaint()
        } else if (
          !svg.contains(target) &&
          name === 'style' &&
          motionOnly(entry.oldValue, target.getAttribute('style'))
        ) {
          for (const owner of groups.values()) {
            owner.moved = true
            owner.fresh = false
          }
        } else if (
          target === scene &&
          (name === 'data-trail-rendered' || name === 'data-trail-visible')
        )
          schedule()
        else if (!svg.contains(target) && !restyle(name)) repaint()
        if (!svg.contains(target)) continue
        if (target === svg && (name === 'viewBox' || name === 'style')) {
          root.moved = true
          continue
        }
        if (target.closest('defs')) {
          touchDefinition(target)
          continue
        }
        const known = groups.get(target)
        if (
          name === 'transform' ||
          (name === 'style' && motionOnly(entry.oldValue, target.getAttribute('style')))
        ) {
          if (name === 'transform' && target.parentElement) invalidateBox(target.parentElement)
          // Scripted motion has the same texture boundary as CSS motion. Waiting for repeated
          // repaints makes the first gust split and reraster each crown several times.
          if (known) {
            known.moved = true
            if (name === 'style') known.fresh = false
          } else promote(target, 'hot')
          continue
        }
        invalidateBox(target)
        if (known && (name === 'mask' || name === 'clip-path' || name === 'class'))
          known.structure = true
        if (name === 'mask') promote(target)
        touch(target)
        continue
      }
      changed = true
      if (target.closest('defs')) {
        touchDefinition(target)
        continue
      }
      if (entry.type === 'childList') {
        invalidateBox(target)
        for (const node of entry.addedNodes) {
          if (!is(node, 'Element')) continue
          discover(node)
          const movedGroup = groups.get(node)
          if (movedGroup) movedGroup.moved = true
        }
        const known = groups.get(target)
        const leaf = known?.items[0]?.kind === 'run' && known.items[0].nodes[0] === target
        if (known && !leaf) {
          known.structure = true
          continue
        }
      }
      touch(target)
    }
    if (changed) {
      shared.initialAnimations = null
      schedule()
    }
  }
  return { readSelectors, record }
}
export type Mutations = ReturnType<typeof createMutations>
