export const UNIVERSITY_ROOT = 'thoughts/university/'
export const UNIVERSITY_SLUG = 'thoughts/university/index'

/** `thoughts/university/<term>/<course>/index` */
export const universityHomeRe = /^thoughts\/university\/([^/]+)\/([^/]+)\/index$/

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
