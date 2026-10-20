import { sluggify } from './path'

export const UNIVERSITY_ROOT = 'thoughts/university/'
export const UNIVERSITY_SLUG = 'thoughts/university/index'

// `twenty-five-twenty-six` -> `2025–26`
const numberWords: Record<string, number> = { three: 3, four: 4, five: 5, six: 6, seven: 7 }
export function termLabel(dir: string): string {
  const [a, b] = dir.split('-twenty-').map(part => numberWords[part.replace(/^twenty-/, '')])
  return a && b ? `20${20 + a}–${20 + b}` : dir
}

// `sfwr-4ml3` -> `SFWR 4ML3`
export const courseCode = (dir: string): string => dir.replace('-', ' ').toUpperCase()

/** McMaster terms start in January, May and September. */
export function semesterOf(date: unknown): string | undefined {
  const match = typeof date === 'string' ? /^(\d{4})-(\d{2})/.exec(date) : null
  if (!match) return undefined
  const month = Number(match[2])
  const season = month <= 4 ? 'winter' : month <= 8 ? 'spring/summer' : 'fall'
  return `${season} ${match[1]}`
}

/** A page inside a course folder: `thoughts/university/<term>/<course>/<rest>`. */
const courseSlugRe = /^thoughts\/university\/([^/]+)\/([^/]+)\/(.+)$/

export interface UniversityPlace {
  term: string
  course: string
  /** folder path below the course, as slug segments; empty at the course root */
  folders: string[]
  /** a folder page (`…/index`): the course home or one of its subfolders */
  isFolder: boolean
}

export function universityPlaceOf(slug: string): UniversityPlace | null {
  const match = courseSlugRe.exec(slug)
  if (!match) return null
  const [, term, course, rest] = match
  const parts = rest.split('/')
  const isFolder = parts[parts.length - 1] === 'index'
  return { term, course, folders: parts.slice(0, -1), isFolder }
}

let folderNames: { paths: readonly string[]; names: Map<string, string> } | undefined

/** Source names of university folders by slug (`…/05-Construction-of-a-Parser` → `05 Construction of a Parser`). */
export function universityFolderName(paths: readonly string[], folderSlug: string): string {
  if (folderNames?.paths !== paths) {
    const names = new Map<string, string>()
    for (const fp of paths) {
      const rel = fp.replace(/^content\//, '')
      if (!rel.startsWith(UNIVERSITY_ROOT)) continue
      const parts = rel.split('/').slice(0, -1)
      for (let i = 1; i <= parts.length; i++) {
        const slug = sluggify(parts.slice(0, i).join('/'))
        if (!names.has(slug)) names.set(slug, parts[i - 1])
      }
    }
    folderNames = { paths, names }
  }
  return folderNames.names.get(folderSlug) ?? folderSlug.slice(folderSlug.lastIndexOf('/') + 1)
}
