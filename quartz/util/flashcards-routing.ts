import type { Root } from 'hast'
import { visit } from 'unist-util-visit'
import { VFile } from 'vfile'
import type { ProcessedContent, QuartzPluginData } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import { clone } from './clone'
import { flashcardsSlug, isFlashcardPath, sourceSlugForDeck } from './flashcards-path'
import {
  FilePath,
  FullSlug,
  SimpleSlug,
  isFullSlug,
  joinSegments,
  pathToRoot,
  simplifySlug,
} from './path'
import { hasHastClass } from './transclude-props'
import { extractWikilinksWithPositions, resolveWikilinkTarget } from './wikilinks'

export interface FlashcardRoute {
  noteSlug: FullSlug
  sourceSlug: FullSlug
  /** Persistent review-state key, independent of the drill's current URL. */
  deckSlug: FullSlug
  default: boolean
  aliases: FullSlug[]
}

function fullSlug(value: string): FullSlug {
  if (!isFullSlug(value)) throw new Error(`Invalid flashcard route: ${value}`)
  return value
}

function routesFor(content: ProcessedContent[]): Map<FullSlug, FlashcardRoute> {
  const notes = new Map(
    content
      .filter(([, file]) => !file.data.flashcards)
      .map(([, file]) => [file.data.slug, file.data]),
  )
  const routes = new Map<FullSlug, FlashcardRoute>()
  const aliases = new Set(content.flatMap(([, file]) => file.data.aliases ?? []))
  for (const [, file] of content) {
    const noteSlug = file.data.flashcards?.sourceSlug
    if (!noteSlug) continue
    const note = notes.get(noteSlug)
    const useDefault = note?.frontmatter?.flashcards?.default === true
    const sourceSlug = useDefault ? fullSlug(joinSegments(noteSlug, 'source')) : noteSlug
    if (useDefault && (notes.has(sourceSlug) || aliases.has(sourceSlug))) {
      throw new Error(`Flashcard source route ${sourceSlug} conflicts with an authored page`)
    }
    if (routes.has(noteSlug)) throw new Error(`Multiple flashcard decks claim ${noteSlug}`)
    routes.set(noteSlug, {
      noteSlug,
      sourceSlug,
      deckSlug: flashcardsSlug(noteSlug),
      default: useDefault,
      aliases: note?.aliases ?? [],
    })
  }
  return routes
}

function targetSlug(slug: string, routes: Map<FullSlug, FlashcardRoute>, source: boolean): string {
  const note =
    (isFullSlug(slug) ? routes.get(slug) : undefined) ??
    (source
      ? [...routes.values()].find(route => route.aliases.some(alias => alias === slug))
      : undefined)
  if (source && note?.default) return note.sourceSlug
  if (slug.endsWith('/flashcards')) {
    const noteSlug = slug.slice(0, -'/flashcards'.length)
    const deck = isFullSlug(noteSlug) ? routes.get(noteSlug) : undefined
    if (deck?.default) return deck.noteSlug
  }
  return slug
}

function routeTree(
  tree: Root,
  from: FullSlug,
  to: FullSlug,
  routes: Map<FullSlug, FlashcardRoute>,
  links: readonly SimpleSlug[],
  baseUrl?: string,
  deck = false,
): { tree: Root; links: SimpleSlug[] } {
  const result: Root = clone(tree)
  const targets = new Map<SimpleSlug, Set<SimpleSlug>>()
  const origin = `https://${baseUrl ?? 'example.com'}`
  const base = new URL(`/${from}`, origin)
  visit(result, 'element', (node, _index, parent) => {
    const transclude =
      hasHastClass(node, 'transclude') ||
      (parent?.type === 'element' && hasHastClass(parent, 'transclude'))
    for (const attr of ['href', 'src', 'dataSrc', 'dataCanvasFile']) {
      const value = node.properties[attr]
      if (typeof value !== 'string' || !value || value.startsWith('#')) continue
      let url: URL
      try {
        url = new URL(value, base)
      } catch {
        continue
      }
      if (deck && !['http:', 'https:', 'mailto:', 'tel:', 'data:'].includes(url.protocol)) {
        node.properties[attr] = ''
        continue
      }
      if (url.origin !== base.origin) continue
      let pathname: string
      try {
        pathname = decodeURI(url.pathname).replace(/^\//, '')
      } catch {
        continue
      }
      const extension = pathname.endsWith('.md') ? '.md' : pathname.endsWith('.html') ? '.html' : ''
      const slug = extension ? pathname.slice(0, -extension.length) : pathname
      const mapped = targetSlug(slug, routes, Boolean(url.hash) || transclude)
      if (mapped === slug && from === to) continue
      const destination = `${mapped}${extension}${url.search}${url.hash}`
      node.properties[attr] = value.startsWith('/')
        ? `/${destination}`
        : /^https?:/.test(value)
          ? `${base.origin}/${destination}`
          : joinSegments(pathToRoot(to), destination)
    }
    for (const attr of ['dataSlug', 'data-slug', 'dataUrl', 'data-url']) {
      const value = node.properties[attr]
      if (typeof value !== 'string') continue
      const mapped = targetSlug(
        value,
        routes,
        transclude ||
          (typeof node.properties.href === 'string' && node.properties.href.includes('#')),
      )
      node.properties[attr] = mapped
      if (
        (attr === 'dataSlug' || attr === 'data-slug') &&
        isFullSlug(value) &&
        isFullSlug(mapped)
      ) {
        const before = simplifySlug(value)
        const destinations = targets.get(before) ?? new Set<SimpleSlug>()
        destinations.add(simplifySlug(mapped))
        targets.set(before, destinations)
      }
    }
  })
  return {
    tree: result,
    links: [
      ...new Set(
        links.flatMap(link => {
          const rewritten = targets.get(link)
          if (rewritten) return [...rewritten]
          const mapped = targetSlug(link, routes, false)
          return [isFullSlug(mapped) ? simplifySlug(mapped) : link]
        }),
      ),
    ],
  }
}

function routeMarkdown(
  text: string | undefined,
  routes: Map<FullSlug, FlashcardRoute>,
  original: FullSlug,
): string | undefined {
  if (text === undefined) return undefined
  for (const range of extractWikilinksWithPositions(text).reverse()) {
    const target = resolveWikilinkTarget(range.wikilink, original)
    if (!target) continue
    const slug = isFlashcardPath(target.slug)
      ? flashcardsSlug(sourceSlugForDeck(target.slug))
      : target.slug
    const mapped = targetSlug(slug, routes, Boolean(range.wikilink.embed || target.anchor))
    if (mapped === target.slug) continue
    text = `${text.slice(0, range.start)}<ref slug="${mapped}${target.anchor ?? ''}">${text.slice(range.end)}`
  }
  return text.replace(
    /(<ref slug=["']?)([^"'>\s]+)(["']?)( transclude)?>/g,
    (_match, prefix: string, target: string, quote: string, transclude: string | undefined) => {
      const hash = target.indexOf('#')
      const slug = hash < 0 ? target : target.slice(0, hash)
      return `${prefix}${targetSlug(slug, routes, hash >= 0 || transclude !== undefined)}${hash < 0 ? '' : target.slice(hash)}${quote}>`
    },
  )
}

const pageCache = new WeakMap<
  VFile,
  { signature: string; tree: Root; note?: QuartzPluginData; page: ProcessedContent }
>()

/** Project parsed files into public pages without changing parser caches or authored identities. */
export function flashcardPages(content: ProcessedContent[], baseUrl?: string): ProcessedContent[] {
  const routes = routesFor(content)
  if (routes.size === 0) return content
  const notes = new Map(
    content
      .filter(([, file]) => !file.data.flashcards)
      .map(([, file]) => [file.data.slug, file.data]),
  )
  const hasDefaults = [...routes.values()].some(route => route.default)
  const signature = `${baseUrl}:${JSON.stringify([...routes].sort(([a], [b]) => a.localeCompare(b)))}`
  return content.map(([tree, file]): ProcessedContent => {
    const original = file.data.slug
    if (!original) return [tree, file]
    const deck = file.data.flashcards
    const route = routes.get(deck?.sourceSlug ?? original)
    const slug = route
      ? deck
        ? route.default
          ? route.noteSlug
          : route.deckSlug
        : route.sourceSlug
      : original
    if (!route && !hasDefaults) return [tree, file]
    const note = route ? notes.get(route.noteSlug) : undefined
    const cached = pageCache.get(file)
    if (cached?.signature === signature && cached.tree === tree && cached.note === note)
      return cached.page
    const data: QuartzPluginData = { ...file.data, slug }
    if (route) data.flashcardRoute = route
    if (deck && route) {
      data.frontmatter = {
        ...note?.frontmatter,
        ...file.data.frontmatter,
        title: note?.frontmatter?.title ?? file.data.frontmatter?.title ?? 'flashcards',
        pageLayout: 'flashcards',
      }
      if (note?.frontmatter?.protected === true) {
        data.frontmatter.protected = true
        data.protectedPassword = note.protectedPassword
      }
      data.rawDescription = note?.rawDescription ?? data.rawDescription
      data.description = note?.description ?? data.description
      data.dates = note?.dates ?? data.dates
      data.aliases = [
        ...new Set([
          ...(file.data.aliases ?? []),
          ...(route.default ? [...(note?.aliases ?? []), route.deckSlug] : []),
        ]),
      ]
      data.llmsText = `# ${data.frontmatter.title}\n\n[Source note](/${route.sourceSlug})\n\n${file.data.llmsText ?? ''}`
    } else if (route?.default) {
      data.aliases = []
      if (data.frontmatter) data.frontmatter = { ...data.frontmatter, aliases: [], permalinks: [] }
    }
    if (slug !== original && data.frontmatter?.socialImage) {
      data.frontmatter = {
        ...data.frontmatter,
        socialImage: new URL(
          data.frontmatter.socialImage,
          `https://${baseUrl ?? 'example.com'}/${original}`,
        ).href,
      }
    }
    data.llmsText = routeMarkdown(
      typeof data.llmsText === 'string' ? data.llmsText : undefined,
      routes,
      original,
    )
    const { tree: nextTree, links } = routeTree(
      tree,
      original,
      slug,
      routes,
      data.links ?? [],
      baseUrl,
      Boolean(deck),
    )
    data.links = links
    if (route && deck) {
      visit(nextTree, 'element', node => {
        if (node.properties.dataCardId || node.properties['data-card-id'])
          node.properties.dataSource = route.sourceSlug
      })
    }
    data.htmlAst = nextTree
    const page: ProcessedContent = [
      nextTree,
      new VFile({ path: file.path, value: file.value, data }),
    ]
    pageCache.set(file, { signature, tree, note, page })
    return page
  })
}

/** Expand dependent note/deck changes and preserve removed routes for emitter cleanup. */
export function flashcardPageChanges(
  content: ProcessedContent[],
  events: ChangeEvent[],
  baseUrl?: string,
): { content: ProcessedContent[]; changes: ChangeEvent[]; removed: FullSlug[] } {
  const previousByPath = new Map(content.map(entry => [entry[1].data.filePath, entry]))
  for (const event of events) {
    const file = event.previousFile ?? (event.type === 'delete' ? event.file : undefined)
    if (file)
      previousByPath.set(file.data.filePath, [
        file.data.htmlAst ?? { type: 'root', children: [] },
        file,
      ])
    else if (event.type === 'add' && event.file) previousByPath.delete(event.file.data.filePath)
  }
  const previousContent = [...previousByPath.values()]
  const current = flashcardPages(content, baseUrl)
  const previous = flashcardPages(previousContent, baseUrl)
  const before = new Map(previous.map(([, file]) => [file.data.filePath, file]))
  const after = new Map(current.map(([, file]) => [file.data.filePath, file]))
  const defaults = (entries: ProcessedContent[]) =>
    JSON.stringify(
      [...routesFor(entries)]
        .filter(([, route]) => route.default)
        .sort(([a], [b]) => a.localeCompare(b)),
    )
  const routesChanged = defaults(content) !== defaults(previousContent)
  const affected = new Set<FilePath | undefined>(
    events.flatMap(event => [event.file?.data.filePath, event.previousFile?.data.filePath]),
  )
  const affectedNotes = new Set(
    events
      .flatMap(event => [event.file, event.previousFile])
      .flatMap(file => (file ? [file.data.flashcards?.sourceSlug ?? file.data.slug] : [])),
  )
  const changes: ChangeEvent[] = events.filter(event => !event.file)
  const removed: FullSlug[] = []
  const currentSlugs = new Set(current.map(([, file]) => file.data.slug))
  for (const [fp, file] of before) {
    if (file.data.slug && !currentSlugs.has(file.data.slug)) {
      removed.push(file.data.slug)
    }
    if (!after.has(fp) || (file.data.slug && !currentSlugs.has(file.data.slug))) {
      changes.push({ type: 'delete', path: file.data.relativePath ?? fp!, file })
    }
  }
  for (const [fp, file] of after) {
    const old = before.get(fp)
    if (
      !routesChanged &&
      !affected.has(fp) &&
      !affectedNotes.has(file.data.flashcardRoute?.noteSlug)
    )
      continue
    changes.push({
      type: old ? 'change' : 'add',
      path: file.data.relativePath ?? fp!,
      file,
      previousFile: old,
    })
  }
  return { content: current, changes, removed }
}

declare module 'vfile' {
  interface DataMap {
    flashcardRoute?: FlashcardRoute
  }
}
