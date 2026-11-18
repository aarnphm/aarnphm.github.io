import type { SimulationLinkDatum, SimulationNodeDatum } from 'd3'
import type { ContentDetails } from '../../../plugins/emitters/contentIndex'
import { isFullSlug, simplifySlug, type SimpleSlug } from '../../../util/path'
import { canonicalTag, canonicalTagRoute, tagLabels } from '../../../util/tags'

export type NodeData = SimulationNodeDatum & {
  id: SimpleSlug
  text: string
  tags: string[]
  radius: number
}
export type LinkData = SimulationLinkDatum<NodeData> & { source: NodeData; target: NodeData }
type Options = { depth: number; showTags: boolean; removeTags: string[] }

export function prepareGraphData(
  index: Record<string, ContentDetails>,
  slug: SimpleSlug,
  options: Options,
): { nodes: NodeData[]; links: LinkData[] } {
  const data = new Map<SimpleSlug, ContentDetails>()
  for (const [key, details] of Object.entries(index)) {
    const canonical = canonicalTagRoute(key)
    if (isFullSlug(canonical)) data.set(simplifySlug(canonical), details)
  }
  const labels = tagLabels([...data.values()])
  const removed = new Set(options.removeTags.map(canonicalTag))
  const links: { source: SimpleSlug; target: SimpleSlug }[] = []
  const tags = new Set<SimpleSlug>()
  const seenLinks = new Set<string>()
  const adjacency = new Map<SimpleSlug, Set<SimpleSlug>>()
  function addLink(source: SimpleSlug, target: SimpleSlug) {
    const key = `${source}\u0000${target}`
    if (seenLinks.has(key)) return
    seenLinks.add(key)
    links.push({ source, target })
    for (const [from, to] of [
      [source, target],
      [target, source],
    ]) {
      let neighbors = adjacency.get(from)
      if (!neighbors) {
        neighbors = new Set()
        adjacency.set(from, neighbors)
      }
      neighbors.add(to)
    }
  }
  for (const [source, details] of data) {
    for (const destination of details.links ?? []) {
      const canonical = canonicalTagRoute(destination)
      const target = isFullSlug(canonical) ? simplifySlug(canonical) : destination
      if (data.has(target)) addLink(source, target)
    }
    if (!options.showTags) continue
    for (const tag of new Set(details.tags.map(canonicalTag))) {
      if (removed.has(tag)) continue
      const key = `tags/${tag}`
      if (!isFullSlug(key)) continue
      const target = simplifySlug(key)
      tags.add(target)
      addLink(source, target)
    }
  }
  const neighbourhood = new Set<SimpleSlug>()
  if (options.depth < 0) {
    for (const id of data.keys()) neighbourhood.add(id)
    for (const id of tags) neighbourhood.add(id)
  } else {
    const canonical = canonicalTagRoute(slug)
    const origin = isFullSlug(canonical) ? simplifySlug(canonical) : slug
    const pending: { id: SimpleSlug; distance: number }[] = [{ id: origin, distance: 0 }]
    const scheduled = new Set([origin])
    for (let cursor = 0; cursor < pending.length; cursor++) {
      const { id, distance } = pending[cursor]
      neighbourhood.add(id)
      if (distance >= options.depth) continue
      for (const neighbor of adjacency.get(id) ?? []) {
        if (!scheduled.has(neighbor)) {
          scheduled.add(neighbor)
          pending.push({ id: neighbor, distance: distance + 1 })
        }
      }
    }
  }
  const nodes: NodeData[] = [...neighbourhood].map(id => ({
    id,
    text: id.startsWith('tags/')
      ? `#${labels[id.slice(5)] ?? id.slice(5)}`
      : (data.get(id)?.title ?? id),
    tags: [...new Set((data.get(id)?.tags ?? []).map(canonicalTag))],
    radius: 2,
  }))
  const byId = new Map(nodes.map(node => [node.id, node]))
  const selected: LinkData[] = []
  const degrees = new Map<SimpleSlug, number>()
  for (const link of links) {
    const source = byId.get(link.source)
    const target = byId.get(link.target)
    if (!source || !target) continue
    selected.push({ source, target })
    degrees.set(source.id, (degrees.get(source.id) ?? 0) + 1)
    if (target.id !== source.id) degrees.set(target.id, (degrees.get(target.id) ?? 0) + 1)
  }
  for (const node of nodes) node.radius = 2 + Math.sqrt(degrees.get(node.id) ?? 0)
  return { nodes, links: selected }
}
