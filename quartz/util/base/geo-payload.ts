import type { QuartzPluginData } from '../../plugins/vfile'
import type { BaseGeoMember, BaseGeoPayload } from './geo-types'
import { simplifySlug } from '../path'

type GeoConfig = { views?: { type?: string; coordinates?: string }[] } | undefined

function fieldString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function fieldType(value: unknown): string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  return value.filter((item): item is string => typeof item === 'string').join(' ')
}

function coordinate(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string' || value.trim().length === 0) return undefined
  const parsed = Number(value.trim())
  return Number.isFinite(parsed) ? parsed : undefined
}

function pin(value: unknown): { lat: number; lon: number } | undefined {
  if (!Array.isArray(value) || value.length !== 2) return undefined
  const lat = coordinate(value[0])
  const lon = coordinate(value[1])
  if (lat === undefined || lon === undefined) return undefined
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return undefined
  return { lat, lon }
}

function searchText(title: string, description: string, address: string, type: string): string {
  return [title, description, address, type].join(' ').replace(/\s+/g, ' ').trim().toLowerCase()
}

export function baseDeclaresCoordinates(config: GeoConfig): boolean {
  return (
    config?.views?.some(
      view =>
        view.type === 'map' ||
        (typeof view.coordinates === 'string' && view.coordinates.trim().length > 0),
    ) ?? false
  )
}

export function buildBaseGeoPayload(
  config: GeoConfig,
  files: QuartzPluginData[],
  memberSlugs: ReadonlySet<string>,
): BaseGeoPayload | null {
  if (!baseDeclaresCoordinates(config)) return null
  const members: BaseGeoMember[] = []
  for (const file of files) {
    if (!file.slug || !memberSlugs.has(simplifySlug(file.slug))) continue
    const frontmatter = file.frontmatter
    const tail = file.slug.split('/').pop() || file.slug
    const title =
      frontmatter && typeof frontmatter.title === 'string' && frontmatter.title.length > 0
        ? frontmatter.title
        : tail
    const description = fieldString(frontmatter?.description)
    const address = fieldString(frontmatter?.address)
    const type = fieldType(frontmatter?.type)
    const coordinates = pin(frontmatter?.coordinates)
    const member: BaseGeoMember = {
      slug: file.slug,
      title,
      searchText: searchText(title, description, address, type),
      fields: { title, type, description, address },
    }
    if (coordinates) {
      member.lat = coordinates.lat
      member.lon = coordinates.lon
    }
    members.push(member)
  }
  return { version: 1, members }
}
