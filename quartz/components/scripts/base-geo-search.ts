import type { BaseGeoPayload } from '../../util/base/geo-types'
import { lexicalSlugs, resolveBaseQuery } from './base-geo-client'
import { isSemanticCancellation } from './semantic.inline'

type SearchMode = 'lexical' | 'semantic'

export interface GeoSearchHost {
  query(): string
  payload: BaseGeoPayload
  mode(): SearchMode
  bump(): number
  current(): number
  active(): boolean
  notice(value: string): void
  status(value: string): void
  cancelEncoder(): void
  releaseEncoder(): void
  setSemanticState(state: 'idle' | 'loading' | 'ready'): void
  showAll(): void
  hideAll(): void
  applySlugs(slugs: string[], distances: Record<string, number>, distanceLabels: boolean): void
  applySemantic(scores: Map<string, number>): void
  dispatch(slugs: string[] | null, distances: Record<string, number>, notice: string): void
  ensureSemantic(seq: number): Promise<string[] | null>
  search(residual: string, limit: number): Promise<{ id: number; score: number }[]>
  semanticK: number
  semanticTail: number
}

export function formatStraightLine(meters: number): string {
  return meters >= 1000
    ? `${(meters / 1000).toFixed(1)} km · straight-line`
    : `${Math.round(meters)} m · straight-line`
}

export function showDistanceLabel(
  labels: Map<HTMLElement, { label: HTMLElement; title: HTMLElement }>,
  item: HTMLElement,
  meters: number,
): void {
  const link =
    item.querySelector<HTMLElement>('.base-card-title-link') ??
    item.querySelector<HTMLElement>('a[data-slug]')
  if (!link) return
  let elements = labels.get(item)
  if (!elements) {
    const title = document.createElement('div')
    title.className = 'base-semantic-title'
    const label = document.createElement('span')
    label.className = 'base-semantic-score'
    link.before(title)
    title.append(link, label)
    elements = { label, title }
    labels.set(item, elements)
  }
  const text = formatStraightLine(meters)
  elements.label.textContent = text
  elements.label.title = text
}

export interface SlugListTarget {
  items: HTMLElement[]
  slugOf(item: HTMLElement): string | undefined
  reorderable(item: HTMLElement): boolean
  clearScore(item: HTMLElement): void
  showDistance(item: HTMLElement, meters: number): void
  bySibling(nodes: Iterable<HTMLElement>): Map<HTMLElement, HTMLElement[]>
  collapseGroups(): void
  setFiltered(value: boolean): void
  setReordered(value: boolean): void
}

export function applyMemberList(
  target: SlugListTarget,
  slugs: string[],
  distances: Record<string, number>,
  distanceLabels: boolean,
): void {
  target.setFiltered(true)
  target.setReordered(slugs.length > 0)
  const slugSet = new Set(slugs)
  const rank = new Map(slugs.map((slug, index) => [slug, index]))
  const ranked: HTMLElement[] = []
  for (const item of target.items) {
    const slug = target.slugOf(item)
    const show = slug !== undefined && slugSet.has(slug)
    item.style.display = show ? '' : 'none'
    if (!show || slug === undefined) {
      target.clearScore(item)
      continue
    }
    if (distanceLabels && distances[slug] !== undefined) {
      delete item.dataset.semanticScore
      target.showDistance(item, distances[slug])
    } else {
      target.clearScore(item)
    }
    if (target.reorderable(item)) ranked.push(item)
  }
  const rankOf = (item: HTMLElement) => rank.get(target.slugOf(item)!) ?? 0
  for (const [parent, siblings] of target.bySibling(ranked)) {
    siblings.sort((left, right) => rankOf(left) - rankOf(right))
    parent.prepend(...siblings)
  }
  target.collapseGroups()
}

export function distancesForSlugs(
  slugs: string[],
  distances: Record<string, number>,
): Record<string, number> {
  const picked: Record<string, number> = {}
  for (const slug of slugs) {
    const meters = distances[slug]
    if (meters !== undefined) picked[slug] = meters
  }
  return picked
}

function spatialHits(
  resolved: ReturnType<typeof resolveBaseQuery>,
): { slug: string; distanceM: number }[] {
  if (resolved.kind !== 'spatial') return []
  return resolved.hits.filter(hit => Number.isFinite(hit.distanceM))
}

export async function runBaseGeoSearch(host: GeoSearchHost): Promise<void> {
  const raw = host.query()
  const seq = host.bump()
  host.notice('')
  host.status('')

  if (raw.length === 0) {
    host.releaseEncoder()
    host.setSemanticState('idle')
    host.showAll()
    host.dispatch(null, {}, '')
    return
  }

  const resolved = resolveBaseQuery(raw, host.payload)
  if (resolved.kind === 'blocked') {
    host.cancelEncoder()
    host.hideAll()
    host.notice(resolved.notice)
    host.status(resolved.notice)
    host.dispatch([], {}, resolved.notice)
    return
  }

  const hits = spatialHits(resolved)
  const distances: Record<string, number> = {}
  for (const hit of hits) distances[hit.slug] = hit.distanceM
  const residual = resolved.residual

  if (resolved.kind === 'spatial' && residual.length === 0) {
    host.cancelEncoder()
    host.releaseEncoder()
    host.setSemanticState('idle')
    const sorted = [...hits].sort((left, right) => left.distanceM - right.distanceM)
    const slugs = sorted.map(hit => hit.slug)
    const distMap = Object.fromEntries(sorted.map(hit => [hit.slug, hit.distanceM]))
    host.applySlugs(slugs, distMap, true)
    host.dispatch(slugs, distMap, '')
    return
  }

  const hitSet = resolved.kind === 'spatial' ? new Set(hits.map(hit => hit.slug)) : null

  if (host.mode() === 'lexical') {
    host.cancelEncoder()
    host.releaseEncoder()
    host.setSemanticState('idle')
    const slugs = lexicalSlugs(host.payload, residual, hitSet)
    host.applySlugs(slugs, distances, false)
    host.dispatch(slugs, distancesForSlugs(slugs, distances), '')
    return
  }

  if (residual.length === 0) return

  host.setSemanticState('loading')
  try {
    const ids = await host.ensureSemantic(seq)
    if (seq !== host.current()) return
    if (!ids) {
      const slugs = lexicalSlugs(host.payload, residual, hitSet)
      host.applySlugs(slugs, distances, false)
      host.dispatch(slugs, distancesForSlugs(slugs, distances), '')
      return
    }
    try {
      const searchLimit = hitSet ? ids.length : host.semanticK
      const semantic = await host.search(residual, searchLimit)
      if (seq !== host.current()) return
      const scores = new Map<string, number>()
      if (hitSet) {
        const survivors: { slug: string; score: number }[] = []
        for (const hit of semantic) {
          const slug = ids[hit.id]
          if (slug && hitSet.has(slug) && Number.isFinite(hit.score)) {
            survivors.push({ slug, score: hit.score })
          }
        }
        survivors.sort((left, right) => right.score - left.score)
        const best = survivors[0]?.score ?? 0
        const floor = Math.max(best - host.semanticTail, 0.6)
        for (const entry of survivors) {
          if (entry.score < floor || scores.size >= host.semanticK) break
          scores.set(entry.slug, entry.score)
        }
      } else {
        const floor = (semantic[0]?.score ?? 0) - host.semanticTail
        for (const hit of semantic) {
          const slug = ids[hit.id]
          if (slug && Number.isFinite(hit.score) && hit.score >= floor && !scores.has(slug)) {
            scores.set(slug, hit.score)
          }
        }
      }
      host.applySemantic(scores)
      const slugs = [...scores.keys()]
      host.dispatch(slugs, distancesForSlugs(slugs, distances), '')
      host.releaseEncoder()
    } catch (err) {
      if (isSemanticCancellation(err)) return
      if (seq !== host.current()) return
      const slugs = lexicalSlugs(host.payload, residual, hitSet)
      host.applySlugs(slugs, hitSet ? distances : {}, false)
      host.dispatch(slugs, hitSet ? distancesForSlugs(slugs, distances) : {}, '')
    }
  } finally {
    if (host.active() && seq === host.current()) host.setSemanticState('ready')
  }
}
