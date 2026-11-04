/** A search query that names a site path: `/embeddings`, `path:/thoughts/`, `/posts/25#notes`. */
export interface PathQuery {
  /** Path without the leading slash; keeps a trailing slash when one was typed. */
  path: string
  /** `?…` and/or `#…` carried onto the destination unchanged. */
  suffix: string
}

export interface PathPage {
  slug: string
  title?: string
}

export interface PathEntry {
  /** Route without slashes at either end; '' is the site root. */
  path: string
  slug?: string
  title?: string
  /** Pages strictly below this path. */
  pages: number
}

export interface PathTree {
  entries: Map<string, PathEntry>
  children: Map<string, PathEntry[]>
  /** Lowercased path -> canonical path, so `/LW` finds `thoughts/LW`. */
  folded: Map<string, string>
}

export interface PathMatches {
  exact?: PathEntry
  items: PathEntry[]
}

export function parsePathQuery(raw: string): PathQuery | null {
  let value = raw.trim()
  const prefixed = /^path:/i.exec(value)
  if (prefixed) value = value.slice(prefixed[0].length).trim()
  if (!value.startsWith('/') || /\s/.test(value)) return null
  const cut = value.search(/[?#]/)
  const head = cut === -1 ? value : value.slice(0, cut)
  return {
    path: head.replace(/\/{2,}/g, '/').replace(/^\//, ''),
    suffix: cut === -1 ? '' : value.slice(cut),
  }
}

/** `thoughts/index` -> `thoughts`, `index` -> ''. */
export function routePath(slug: string): string {
  if (slug === 'index') return ''
  return slug.endsWith('/index') ? slug.slice(0, -'/index'.length) : slug
}

function parentPath(path: string) {
  const cut = path.lastIndexOf('/')
  return cut === -1 ? '' : path.slice(0, cut)
}

function lastSegment(path: string) {
  return path.slice(path.lastIndexOf('/') + 1)
}

export function buildPathTree(pages: Iterable<PathPage>): PathTree {
  const entries = new Map<string, PathEntry>([['', { path: '', pages: 0 }]])
  const ensure = (path: string) => {
    let entry = entries.get(path)
    if (!entry) {
      entry = { path, pages: 0 }
      entries.set(path, entry)
    }
    return entry
  }
  for (const page of pages) {
    const path = routePath(page.slug)
    const entry = ensure(path)
    entry.slug = page.slug
    entry.title = page.title
    for (let up = path; up !== ''; ) {
      up = parentPath(up)
      ensure(up).pages += 1
    }
  }
  const children = new Map<string, PathEntry[]>()
  const folded = new Map<string, string>()
  for (const entry of entries.values()) {
    folded.set(entry.path.toLowerCase(), entry.path)
    if (entry.path === '') continue
    const parent = parentPath(entry.path)
    const list = children.get(parent)
    if (list) list.push(entry)
    else children.set(parent, [entry])
  }
  return { entries, children, folded }
}

/** 0 prefix, 1 substring, 2 subsequence (three letters or more), -1 no match. */
function segmentScore(segment: string, typed: string) {
  if (!typed) return 0
  const name = segment.toLowerCase()
  if (name.startsWith(typed)) return 0
  if (name.includes(typed)) return 1
  if (typed.length < 3) return -1
  let at = 0
  for (const character of name) if (character === typed[at]) at++
  return at === typed.length ? 2 : -1
}

function byFolderThenName(a: PathEntry, b: PathEntry) {
  return (
    Number(b.pages > 0) - Number(a.pages > 0) ||
    lastSegment(a.path).localeCompare(lastSegment(b.path), undefined, { sensitivity: 'base' })
  )
}

/**
 * Children of the typed directory whose next segment matches what follows the last slash,
 * then deeper paths under that directory whose final segment matches it, then loose
 * subsequence hits among the children.
 */
export function matchPaths(tree: PathTree, typed: string, deepLimit = 200): PathMatches {
  const trailing = typed.endsWith('/')
  const clean = typed.replace(/\/+$/, '')
  const exactPath = tree.folded.get(clean.toLowerCase())
  const exactEntry = exactPath === undefined ? undefined : tree.entries.get(exactPath)
  const exact = exactEntry?.slug !== undefined ? exactEntry : undefined

  const dirTyped = trailing ? clean : parentPath(clean)
  const leaf = trailing ? '' : lastSegment(clean).toLowerCase()
  const dir = tree.folded.get(dirTyped.toLowerCase())
  if (dir === undefined) {
    const needle = clean.toLowerCase()
    const items = [...tree.entries.values()]
      .filter(entry => entry.path && entry.path.toLowerCase().includes(needle))
      .sort((a, b) => a.path.split('/').length - b.path.split('/').length || byFolderThenName(a, b))
      .slice(0, deepLimit)
    return { exact, items }
  }

  const seen = new Set<string>(exact ? [exact.path] : [])
  const direct = (tree.children.get(dir) ?? [])
    .flatMap(entry => {
      if (seen.has(entry.path)) return []
      const score = segmentScore(lastSegment(entry.path), leaf)
      return score === -1 ? [] : [{ entry, score }]
    })
    .sort((a, b) => a.score - b.score || byFolderThenName(a.entry, b.entry))
  for (const { entry } of direct) seen.add(entry.path)
  if (!leaf) return { exact, items: direct.map(({ entry }) => entry) }

  const prefix = dir ? `${dir}/` : ''
  const deep: { entry: PathEntry; score: number; depth: number }[] = []
  for (const entry of tree.entries.values()) {
    if (!entry.path || seen.has(entry.path) || !entry.path.startsWith(prefix)) continue
    const score = segmentScore(lastSegment(entry.path), leaf)
    if (score === -1 || score === 2) continue
    deep.push({ entry, score, depth: entry.path.split('/').length })
  }
  deep.sort((a, b) => a.score - b.score || a.depth - b.depth || byFolderThenName(a.entry, b.entry))
  // A loose subsequence hit on a direct child ranks below real matches further down.
  return {
    exact,
    items: [
      ...direct.filter(({ score }) => score < 2),
      ...deep.slice(0, deepLimit),
      ...direct.filter(({ score }) => score === 2),
    ].map(({ entry }) => entry),
  }
}

/** What Tab writes into the bar for an entry: folders keep a trailing slash to descend. */
export function completePath(entry: PathEntry, suffix = '') {
  return `/${entry.path}${entry.pages > 0 && !suffix ? '/' : ''}${suffix}`
}
