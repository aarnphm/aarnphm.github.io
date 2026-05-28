import path from 'path'
import { GlobalConfiguration } from '../../cfg'
import { i18n } from '../../i18n'
import { QuartzPluginData } from '../../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { inheritComponentSourceNames } from '../../util/component-source'
import { BuildCtx } from '../../util/ctx'
import { FileTrieNode } from '../../util/fileTrie'
import { htmlToJsx } from '../../util/jsx'
import {
  stripSlashes,
  simplifySlug,
  joinSegments,
  FullSlug,
  slugifyFilePath,
  FilePath,
  SimpleSlug,
  sluggify,
} from '../../util/path'
import { concatenateResources } from '../../util/resources'
import { parseWikilink } from '../../util/wikilinks'
import EvergreenConstructor, { AllTags, EvergreenPermanentNotes } from '../Evergreen'
import PageListConstructor, {
  byDateAndAlphabetical,
  byNaturalSlug,
  byTitleAlphabetical,
  SortFn,
} from '../PageList'
import PageListSearchConstructor from '../PageListSearch'
import style from '../styles/listPage.scss'

interface FolderContentOptions {
  /**
   * Sort function for the pages
   */
  sort?: SortFn
  /**
   * File extensions to include (e.g., [".md", ".pdf", ".ipynb"])
   * If not provided, defaults to showing all files
   */
  include?: (string | RegExp)[]
  /**
   * File extensions to exclude
   * If not provided, no extensions are excluded
   */
  exclude?: (string | RegExp)[]
  lg: string[]
  sm: string[]
  tags: string[]
}

function extensionFilterFn(opts: FolderContentOptions): (filePath: string) => boolean {
  const matchesPattern = (filePath: string, pattern: string | RegExp): boolean => {
    if (pattern instanceof RegExp) {
      return pattern.test(filePath)
    }
    // For string patterns, treat them as exact matches (could be file extensions or exact names)
    if (pattern.startsWith('.')) {
      // If it starts with a dot, treat as extension
      return filePath.toLowerCase().endsWith(pattern.toLowerCase())
    }
    return filePath === pattern
  }

  return (filePath: string): boolean => {
    if (!opts.include && !opts.exclude) return true
    if (opts.exclude?.some(pattern => matchesPattern(filePath, pattern))) return false
    return opts.include?.some(pattern => matchesPattern(filePath, pattern)) ?? true
  }
}

const defaultOptions: FolderContentOptions = {
  include: undefined,
  exclude: undefined,
  lg: [],
  sm: [],
  tags: [],
}

const slugForDirectoryEntry = (fp: FilePath): string => {
  return stripSlashes(slugifyFilePath(fp, path.extname(fp) === '.ipynb'))
}

const Layout = { defn: 'L->EAT', etas: 'L->ET|A', alsp: 'A|L', lovp: 'L' } as const

type FolderLayout = (typeof Layout)[keyof typeof Layout]
type FolderSortMode = 'date' | 'natural' | 'title'

const folderSortAliases: Record<string, FolderSortMode> = {
  alphabetical: 'title',
  alphabetic: 'title',
  title: 'title',
  date: 'date',
  chronological: 'date',
  created: 'date',
  modified: 'date',
  published: 'date',
  natural: 'natural',
  grammar: 'natural',
  grammatical: 'natural',
  lexical: 'natural',
  lexicographic: 'natural',
  path: 'natural',
  file: 'natural',
  filename: 'natural',
}

const parseFolderLayout = (input: string): FolderLayout => {
  if (Array.isArray(input)) return parseFolderLayout(input[0])

  const normalized = input.trim().toUpperCase() as FolderLayout
  const valid = Object.values(Layout)
  return valid.includes(normalized) ? normalized : Layout.defn
}

const parseFolderSortMode = (input: unknown): FolderSortMode | undefined => {
  if (Array.isArray(input)) return parseFolderSortMode(input[0])
  if (typeof input !== 'string') return undefined
  return folderSortAliases[input.trim().toLowerCase()]
}

const sortForFolderMode = (mode: FolderSortMode, cfg: GlobalConfiguration): SortFn => {
  switch (mode) {
    case 'date':
      return byDateAndAlphabetical(cfg)
    case 'natural':
      return byNaturalSlug(cfg)
    case 'title':
      return byTitleAlphabetical(cfg)
  }
}

/**
 * Normalize path entries to slugs, supporting both plain paths and wikilink syntax
 * Examples:
 *   - "thoughts/love" -> "thoughts/love"
 *   - "thoughts/mechanistic interpretability" -> "thoughts/mechanistic-interpretability"
 *   - "[[thoughts/love]]" -> "thoughts/love"
 *   - "[[thoughts/love|Love]]" -> "thoughts/love"
 *   - "[[thoughts/love#section]]" -> "thoughts/love"
 */
const normalizePath = (pathEntry: string): SimpleSlug => {
  const trimmed = pathEntry.trim()

  // Try parsing as wikilink first
  const parsed = parseWikilink(trimmed)
  if (parsed && parsed.target) {
    // Use the target, ignore anchor and alias
    // Slugify to convert spaces to dashes
    const slugified = sluggify(parsed.target)
    return simplifySlug(stripSlashes(slugified) as FullSlug)
  }

  // Fall back to treating as plain path
  // Slugify to convert spaces to dashes (e.g., "mechanistic interpretability" -> "mechanistic-interpretability")
  const slugified = sluggify(trimmed)
  return simplifySlug(stripSlashes(slugified) as FullSlug)
}

type FolderContentIndex = {
  allFiles: QuartzPluginData[]
  allPaths: FilePath[]
  aliases: Set<string>
  descendantsByFolder: Map<string, QuartzPluginData[]>
  firstDescendantByFolderBase: Map<string, QuartzPluginData>
  firstDescendantByFolder: Map<string, QuartzPluginData>
  fullTrie: FileTrieNode<{ slug: string; title: string; filePath: string }>
  immediateMarkdownByFolder: Map<string, QuartzPluginData[]>
  mdBySlug: Map<string, QuartzPluginData>
  navigableFolders: Set<string>
}

function pushMapArray<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const values = map.get(key)
  if (values) {
    values.push(value)
  } else {
    map.set(key, [value])
  }
}

function parentFolders(slug: string): string[] {
  const folders: string[] = []
  let folder = path.posix.dirname(slug)
  while (folder !== '.') {
    folders.push(folder)
    folder = path.posix.dirname(folder)
  }
  return folders
}

const folderPageSourceExtensions = new Set(['.md', '.base', '.canvas', '.ipynb'])

function isFolderPageSourcePath(fp: FilePath): boolean {
  return folderPageSourceExtensions.has(path.extname(fp))
}

function folderPageSourceSlug(fp: FilePath): string {
  return stripSlashes(slugifyFilePath(fp, path.extname(fp) === '.ipynb'))
}

function folderPageAncestors(slug: string): string[] {
  const folders: string[] = []
  let folder = path.posix.dirname(slug)
  while (folder !== '.' && folder !== '/') {
    const simple = stripSlashes(simplifySlug(folder as FullSlug))
    if (simple !== 'tags') folders.push(simple)
    folder = path.posix.dirname(folder)
  }
  return folders
}

function folderBaseKey(folder: string, base: string): string {
  return `${folder}\u0000${base}`
}

function firstByDate(
  files: QuartzPluginData[],
  cfg: GlobalConfiguration,
): QuartzPluginData | undefined {
  return files.length > 0 ? [...files].sort(byDateAndAlphabetical(cfg))[0] : undefined
}

function buildFolderContentIndex(
  ctx: BuildCtx,
  allFiles: QuartzPluginData[],
  cfg: GlobalConfiguration,
): FolderContentIndex {
  const mdBySlug = new Map<string, QuartzPluginData>()
  const descendantsByFolder = new Map<string, QuartzPluginData[]>()
  const descendantsByFolderBase = new Map<string, QuartzPluginData[]>()
  const immediateMarkdownByFolder = new Map<string, QuartzPluginData[]>()
  const aliases = new Set<string>()
  const navigableFolders = new Set<string>()

  for (const file of allFiles) {
    const slug = file.slug
    if (!slug) continue
    for (const folder of folderPageAncestors(slug)) {
      navigableFolders.add(folder)
    }
    const simple = stripSlashes(simplifySlug(slug))
    mdBySlug.set(simple, file)

    const immediateFolder = path.posix.dirname(simple)
    if (immediateFolder !== '.') {
      pushMapArray(immediateMarkdownByFolder, immediateFolder, file)
    }
    const base = path.posix.basename(simple, path.posix.extname(simple))
    for (const folder of parentFolders(simple)) {
      pushMapArray(descendantsByFolder, folder, file)
      pushMapArray(descendantsByFolderBase, folderBaseKey(folder, base), file)
    }

    for (const alias of file.aliases ?? []) {
      aliases.add(stripSlashes(simplifySlug(alias)))
    }
  }

  const firstDescendantByFolder = new Map<string, QuartzPluginData>()
  for (const [folder, files] of descendantsByFolder) {
    const first = firstByDate(files, cfg)
    if (first) firstDescendantByFolder.set(folder, first)
  }
  const firstDescendantByFolderBase = new Map<string, QuartzPluginData>()
  for (const [key, files] of descendantsByFolderBase) {
    const first = firstByDate(files, cfg)
    if (first) firstDescendantByFolderBase.set(key, first)
  }

  const fullTrie = new FileTrieNode<{ slug: string; title: string; filePath: string }>([])
  for (const fp of ctx.allFiles) {
    if (isFolderPageSourcePath(fp)) {
      for (const folder of folderPageAncestors(folderPageSourceSlug(fp))) {
        navigableFolders.add(folder)
      }
    }
    const fileSlug = slugForDirectoryEntry(fp)
    const ext = path.extname(fp)
    const base = path.basename(fp, ext)
    const md = mdBySlug.get(fileSlug)
    fullTrie.add({ slug: fileSlug, title: md?.frontmatter?.title ?? base, filePath: fp })
  }

  return {
    allFiles,
    allPaths: ctx.allFiles,
    aliases,
    descendantsByFolder,
    firstDescendantByFolderBase,
    firstDescendantByFolder,
    fullTrie,
    immediateMarkdownByFolder,
    mdBySlug,
    navigableFolders,
  }
}

export default ((opts?: Partial<FolderContentOptions>) => {
  const options: FolderContentOptions = { ...defaultOptions, ...opts }

  let folderContentIndex: FolderContentIndex | undefined

  const shouldIncludeFile = extensionFilterFn(options)

  // NOTE: we will always add the generated tags "folder" for better distinction
  // Normalize lg/sm paths to support both plain paths and wikilink syntax
  const { tags } = options
  const lg = options.lg.map(normalizePath)
  const sm = options.sm.map(normalizePath)

  const PageList = PageListConstructor({ highlightTags: [...tags] })
  const Evergreen = EvergreenConstructor({ lg, sm, tags })
  const PermanentNotes = EvergreenPermanentNotes({ lg, sm, tags })
  const PageListSearch = PageListSearchConstructor()

  const FolderContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { tree, fileData, allFiles, ctx, cfg } = props
    if (
      !folderContentIndex ||
      folderContentIndex.allFiles !== allFiles ||
      folderContentIndex.allPaths !== ctx.allFiles
    ) {
      folderContentIndex = buildFolderContentIndex(ctx, allFiles, cfg)
    }
    const {
      aliases,
      descendantsByFolder,
      firstDescendantByFolderBase,
      firstDescendantByFolder,
      fullTrie,
      immediateMarkdownByFolder,
      mdBySlug,
      navigableFolders,
    } = folderContentIndex

    const folderSlug = stripSlashes(simplifySlug(fileData.slug!))
    const entries: QuartzPluginData[] = []
    const processed = new Set<string>()

    const folderNode = fullTrie.findNode(folderSlug.split(path.posix.sep))
    const isImagesPath = (slug: string) => slug.split('/').includes('images')

    // Compute a sensible date for the current folder (used as fallback for children)
    const folderIndexMd = mdBySlug.get(folderSlug)
    const filesUnderCurrent = descendantsByFolder.get(folderSlug) ?? []
    const defaultDate = { created: new Date(0), modified: new Date(0), published: new Date(0) }
    const currentFolderDates =
      filesUnderCurrent.length > 0
        ? firstDescendantByFolder.get(folderSlug)?.dates
        : (folderIndexMd?.dates ?? fileData?.dates)

    const pushFileEntry = (fileSlug: string, filePathStr: string) => {
      if (processed.has(fileSlug)) return
      const ext = path.extname(filePathStr)
      const baseFileName = path.basename(filePathStr, ext)
      if (!shouldIncludeFile(filePathStr)) return
      if (isImagesPath(fileSlug)) return

      // If this slug corresponds to a markdown page we know about, just use it directly
      const md = mdBySlug.get(fileSlug)
      if (md) {
        if (md.frontmatter?.noindex === true) return
        // Augment missing dates so PageList can render consistently
        const folderFallback = currentFolderDates || fileData.dates
        const augmentedDates = {
          created: md.dates?.created ?? folderFallback?.created ?? defaultDate.created,
          modified: md.dates?.modified ?? folderFallback?.modified ?? defaultDate.modified,
          published: md.dates?.published ?? folderFallback?.published ?? defaultDate.published,
        }
        entries.push({ ...md, dates: augmentedDates })
        processed.add(fileSlug)
        return
      }

      // Pull dates (prefer markdown companion if present), else fallback to current folder date
      const associatedFirst = firstDescendantByFolderBase.get(
        folderBaseKey(folderSlug, baseFileName),
      )

      const dates = associatedFirst?.dates || currentFolderDates || fileData.dates || defaultDate

      entries.push({
        slug: (fileSlug as FullSlug) ?? (joinSegments(folderSlug, baseFileName) as FullSlug),
        frontmatter: {
          title: mdBySlug.get(fileSlug)?.frontmatter?.title ?? baseFileName,
          tags: [ext.replace('.', '') || 'file'],
          pageLayout: 'default',
        },
        dates,
      })
      processed.add(fileSlug)
    }

    if (folderNode) {
      // Immediate children only: files and subfolders
      const subfolders: FileTrieNode<any>[] = []
      for (const child of folderNode.children) {
        if (child.isFolder) {
          subfolders.push(child)
          continue
        }
        if (!child.data) continue
        const fileSlug = stripSlashes(child.slug)
        const isAlias = aliases.has(stripSlashes(simplifySlug(fileSlug as FullSlug)))
        if (isAlias) continue
        pushFileEntry(fileSlug, child.data.filePath)
      }

      // Add subfolders
      for (const sub of subfolders) {
        // Trie folder slug includes `index` (e.g., a/b/index)
        const subfolderSlugWithIndex = stripSlashes(sub.slug)
        const subfolderSimple = stripSlashes(simplifySlug(sub.slug as FullSlug))
        if (isImagesPath(subfolderSimple) || isImagesPath(subfolderSlugWithIndex)) continue
        if (!navigableFolders.has(subfolderSimple)) continue

        // If there is a markdown index for this folder, rely on that page (avoid duplicate)
        const folderIndex = mdBySlug.get(subfolderSimple)

        // Determine dates from files under the subfolder
        const filesInSubfolder = descendantsByFolder.get(subfolderSimple) ?? []

        const subfolderDates =
          filesInSubfolder.length > 0
            ? firstDescendantByFolder.get(subfolderSimple)?.dates
            : (folderIndex?.dates ?? fileData?.dates)

        // Only generate a synthetic folder entry if no explicit folder index exists
        if (!folderIndex) {
          entries.push({
            // keep `.../index` so it?s treated as a folder in sorting
            slug: subfolderSlugWithIndex as FullSlug,
            frontmatter: {
              title: sub.displayName || sub.slugSegment,
              tags: ['folder'],
              pageLayout: 'default',
            },
            dates: subfolderDates,
          })
          // Mark both forms as processed to prevent any fallback duplication
          processed.add(subfolderSlugWithIndex)
          processed.add(subfolderSimple)
        }
      }
    }

    // Fallback: ensure immediate markdown children are present
    for (const file of immediateMarkdownByFolder.get(folderSlug) ?? []) {
      const fileSlug = stripSlashes(simplifySlug(file.slug!))
      if (!processed.has(fileSlug)) {
        if (file.frontmatter?.noindex === true) continue
        const folderFallback = currentFolderDates || fileData.dates
        const augmentedDates = {
          created: file.dates?.created ?? folderFallback?.created ?? defaultDate.created,
          modified: file.dates?.modified ?? folderFallback?.modified ?? defaultDate.modified,
          published: file.dates?.published ?? folderFallback?.published ?? defaultDate.published,
        }
        entries.push({ ...file, dates: augmentedDates })
        processed.add(fileSlug)
      }
    }

    const layout = parseFolderLayout(
      fileData.frontmatter ? fileData.frontmatter.pageLayout! : 'L->ET|A',
    )

    const cssClasses: string[] = fileData.frontmatter?.cssclasses ?? []
    const baseClassList = ['popover-hint', 'notes-list', 'side-col', ...cssClasses]
    const baseListClass = baseClassList.join(' ')
    const listClassName =
      layout === Layout.etas ? `${baseListClass} folder-layout--list` : baseListClass
    const content = htmlToJsx(fileData.filePath!, tree)

    const sortMode = parseFolderSortMode(
      fileData.frontmatter?.folderSort ??
        fileData.frontmatter?.sort ??
        fileData.frontmatter?.sortBy,
    )
    const sort = sortMode ? sortForFolderMode(sortMode, cfg) : options.sort
    const listProps = { ...props, sort, content, allFiles: entries, vaults: allFiles }

    switch (layout) {
      case Layout.etas:
        return (
          <div class="folder-layout folder-layout--et-a" data-pagelist>
            <section class={listClassName}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
            <div class="notes-evergreen folder-layout--evergreen">
              <PermanentNotes {...listProps} />
              <AllTags {...listProps} opts />
            </div>
            <article class="folder-layout--article">
              {content}
              <p>
                {i18n(cfg.locale).pages.folderContent.itemsUnderFolder({
                  count: listProps.allFiles.length,
                })}
              </p>
            </article>
          </div>
        )

      case Layout.alsp:
        return (
          <div class="folder-layout folder-layout--a-l" data-pagelist>
            <article class="folder-layout--article">
              {content}
              <p>
                {i18n(cfg.locale).pages.folderContent.itemsUnderFolder({
                  count: listProps.allFiles.length,
                })}
              </p>
            </article>
            <section class={listClassName}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
          </div>
        )

      case Layout.lovp:
        return (
          <div class="folder-layout folder-layout--l" data-pagelist>
            <section class={listClassName}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
          </div>
        )

      default:
        return (
          <>
            <section class={baseListClass}>
              <PageListSearch {...props} />
              <PageList {...listProps} />
            </section>
            <aside class="notes-evergreen">
              <Evergreen {...listProps} />
            </aside>
          </>
        )
    }
  }

  FolderContent.css = concatenateResources(style, Evergreen.css, PageListSearch.css)
  FolderContent.sourceNames = inheritComponentSourceNames('FolderContent', [
    Evergreen,
    PageList,
    PageListSearch,
  ])
  FolderContent.afterDOMLoaded = concatenateResources(
    Evergreen.afterDOMLoaded,
    PageListSearch.afterDOMLoaded,
  )

  return FolderContent
}) satisfies QuartzComponentConstructor
