import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import path from "path"
import style from "../styles/listPage.scss"
import PageListConstructor, { byDateAndAlphabetical, SortFn } from "../PageList"
import {
  stripSlashes,
  simplifySlug,
  joinSegments,
  FullSlug,
  slugifyFilePath,
  FilePath,
} from "../../util/path"
import { Root } from "hast"
import { htmlToJsx } from "../../util/jsx"
import { QuartzPluginData } from "../../plugins/vfile"
import EvergreenConstructor from "../Evergreen"
import { FileTrieNode } from "../../util/fileTrie"

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
}

function extensionFilterFn(opts: FolderContentOptions): (filePath: string) => boolean {
  const matchesPattern = (filePath: string, pattern: string | RegExp): boolean => {
    if (pattern instanceof RegExp) {
      return pattern.test(filePath)
    }
    // For string patterns, treat them as exact matches (could be file extensions or exact names)
    if (pattern.startsWith(".")) {
      // If it starts with a dot, treat as extension
      return filePath.toLowerCase().endsWith(pattern.toLowerCase())
    }
    return filePath === pattern
  }

  return (filePath: string): boolean => {
    if (!opts.include && !opts.exclude) return true
    if (opts.exclude?.some((pattern) => matchesPattern(filePath, pattern))) return false
    return opts.include?.some((pattern) => matchesPattern(filePath, pattern)) ?? true
  }
}

const defaultOptions: FolderContentOptions = {
  include: undefined,
  exclude: undefined,
}

export default ((opts?: Partial<FolderContentOptions>) => {
  const options: FolderContentOptions = { ...defaultOptions, ...opts }

  // Trie covering ALL files in content (any extension), built from ctx
  let fullTrie: FileTrieNode<{
    slug: string
    title: string
    filePath: string
  }>

  const shouldIncludeFile = extensionFilterFn(options)

  const tags = ["ml", "interp", "philosophy", "serving", "love", "fiction", "math", "evergreen"]
  // NOTE: we will always add the generated tags "folder" for better distinction
  const PageList = PageListConstructor({ highlightTags: [...tags, "folder"] })
  const Evergreen = EvergreenConstructor({
    lg: ["thoughts/mechanistic-interpretability", "thoughts/vllm"],
    sm: [
      "are.na",
      "thoughts/constrained-decoding",
      "thoughts/LLMs",
      "thoughts/Transformers",
      "thoughts/Camus",
      "thoughts/atelier-with-friends",
      "thoughts/Attention",
      "thoughts/Philosophy-and-Nietzsche",
      "thoughts/ethics",
      "thoughts/Existentialism",
      "thoughts/Scents",
    ],
    tags,
  })

  const FolderContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { tree, fileData, allFiles, ctx, cfg } = props
    // Map markdown/pdf plugin data by slug for metadata lookups
    const mdBySlug = new Map<string, QuartzPluginData>()
    for (const f of allFiles) {
      if (f.slug) mdBySlug.set(stripSlashes(f.slug), f)
    }

    // Initialize a trie with ALL files from the vault (ctx)
    if (!fullTrie) {
      fullTrie = new FileTrieNode([])
      for (const fp of ctx.allFiles) {
        const slug = slugifyFilePath(fp as FilePath)
        const fileSlug = stripSlashes(slug)
        const ext = path.extname(fp)
        const base = path.basename(fp, ext)
        const md = mdBySlug.get(fileSlug)
        fullTrie.add({
          slug: fileSlug,
          title: md?.frontmatter?.title ?? base,
          filePath: fp,
        })
      }
    }

    const folderSlug = stripSlashes(simplifySlug(fileData.slug!))
    const entries: QuartzPluginData[] = []
    const processed = new Set<string>()

    const folderNode = fullTrie.findNode(folderSlug.split(path.posix.sep))
    const isImagesPath = (slug: string) => slug.split("/").includes("images")

    // Compute a sensible date for the current folder (used as fallback for children)
    const folderIndexMd = allFiles.find((f) => stripSlashes(simplifySlug(f.slug!)) === folderSlug)
    const filesUnderCurrent = allFiles.filter((f) =>
      stripSlashes(simplifySlug(f.slug!)).startsWith(`${folderSlug}/`),
    )
    const defaultDate = { created: new Date(0), modified: new Date(0), published: new Date(0) }
    const currentFolderDates =
      filesUnderCurrent.length > 0
        ? filesUnderCurrent.sort(byDateAndAlphabetical(cfg))[0].dates
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
      const associatedFiles = allFiles.filter((f) => {
        const fSlug = stripSlashes(simplifySlug(f.slug!))
        const fBase = path.basename(fSlug, path.extname(fSlug))
        const inFolder = fSlug.startsWith(`${folderSlug}/`)
        return inFolder && fBase === baseFileName
      })

      const sortedFiles = associatedFiles.sort(byDateAndAlphabetical(cfg))
      const dates =
        sortedFiles.length > 0
          ? sortedFiles[0].dates || currentFolderDates || fileData.dates || defaultDate
          : currentFolderDates || fileData.dates || defaultDate

      entries.push({
        slug: (fileSlug as FullSlug) ?? (joinSegments(folderSlug, baseFileName) as FullSlug),
        frontmatter: {
          title: mdBySlug.get(fileSlug)?.frontmatter?.title ?? baseFileName,
          tags: [ext.replace(".", "") || "file"],
          pageLayout: "default",
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
        const isAlias = allFiles.some((f) =>
          f.aliases?.some(
            (alias) => simplifySlug(alias) === stripSlashes(simplifySlug(fileSlug as FullSlug)),
          ),
        )
        if (isAlias) continue
        pushFileEntry(fileSlug, child.data.filePath)
      }

      // Add subfolders
      for (const sub of subfolders) {
        // Trie folder slug includes `index` (e.g., a/b/index)
        const subfolderSlugWithIndex = stripSlashes(sub.slug)
        const subfolderSimple = stripSlashes(simplifySlug(sub.slug as FullSlug))
        if (isImagesPath(subfolderSimple) || isImagesPath(subfolderSlugWithIndex)) continue

        // If there is a markdown index for this folder, rely on that page (avoid duplicate)
        const folderIndex = allFiles.find((f) => {
          const s = stripSlashes(simplifySlug(f.slug!))
          return s === subfolderSimple
        })

        // Determine dates from files under the subfolder
        const filesInSubfolder = allFiles.filter((file) => {
          const s = stripSlashes(simplifySlug(file.slug!))
          return s.startsWith(`${subfolderSimple}/`)
        })

        const subfolderDates =
          filesInSubfolder.length > 0
            ? filesInSubfolder.sort(byDateAndAlphabetical(cfg))[0].dates
            : (folderIndex?.dates ?? fileData?.dates)

        // Only generate a synthetic folder entry if no explicit folder index exists
        if (!folderIndex) {
          entries.push({
            // keep `.../index` so it’s treated as a folder in sorting
            slug: subfolderSlugWithIndex as FullSlug,
            frontmatter: {
              title: sub.displayName || sub.slugSegment,
              tags: ["folder"],
              pageLayout: "default",
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
    for (const file of allFiles) {
      const fileSlug = stripSlashes(simplifySlug(file.slug!))
      if (fileSlug.startsWith(`${folderSlug}/`)) {
        const relativePath = fileSlug.slice(folderSlug.length + 1)
        if (!relativePath.includes("/")) {
          if (!processed.has(fileSlug)) {
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
      }
    }

    const cssClasses: string[] = fileData.frontmatter?.cssclasses ?? []
    const classes = ["popover-hint", "notes-list", "side-col", ...cssClasses].join(" ")
    const content =
      (tree as Root).children.length === 0
        ? fileData.description
        : htmlToJsx(fileData.filePath!, tree)

    const listProps = {
      ...props,
      sort: options.sort,
      content,
      allFiles: entries,
      vaults: allFiles,
    }

    return (
      <>
        <section class={classes}>
          <h3 class="note-title">écriture</h3>
          <PageList {...listProps} />
        </section>
        <aside class="notes-evergreen">
          <Evergreen {...listProps} />
        </aside>
      </>
    )
  }

  FolderContent.css = style + Evergreen.css
  FolderContent.afterDOMLoaded = Evergreen.afterDOMLoaded

  return FolderContent
}) satisfies QuartzComponentConstructor
