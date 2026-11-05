import path from 'path'
import type { GlobalConfiguration } from '../../../cfg'
import type { QuartzPluginData } from '../../../plugins/vfile'
import type { BuildCtx } from '../../../util/ctx'
import { FileTrieNode } from '../../../util/fileTrie'
import { isFlashcardPath } from '../../../util/flashcards-path'
import { folderPageSourceSlug, isFolderPageSourcePath } from '../../../util/folder-page'
import {
  stripSlashes,
  simplifySlug,
  slugifyFilePath,
  type FullSlug,
  type FilePath,
} from '../../../util/path'
import { pdfTitlesFor } from '../../../util/pdf-titles'
import { referenceSourcePath } from '../../../util/references/context'
import { missingReferenceRoute } from '../../../util/references/route'
import { byDateAndAlphabetical } from '../../PageList'

const slugForDirectoryEntry = (fp: FilePath): string => {
  return stripSlashes(slugifyFilePath(fp, path.extname(fp) === '.ipynb'))
}

type FolderContentIndex = {
  allFiles: QuartzPluginData[]
  allPaths: FilePath[]
  aliases: Set<string>
  descendantsByFolder: Map<string, QuartzPluginData[]>
  firstDescendantWithBasename: (folder: string, base: string) => QuartzPluginData | undefined
  firstDescendantByFolder: Map<string, QuartzPluginData>
  fullTrie: FileTrieNode<{ slug: string; title: string; filePath: string }>
  immediateMarkdownByFolder: Map<string, QuartzPluginData[]>
  mdBySlug: Map<string, QuartzPluginData>
  navigableFolders: Set<string>
  pdfTitles: Map<string, string>
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

export function buildFolderContentIndex(
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
  const pdfTitles = pdfTitlesFor(allFiles)

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
      const route = missingReferenceRoute(alias, referenceSourcePath(file))
      if (route) aliases.add(stripSlashes(simplifySlug(route.slug)))
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
    if (isFlashcardPath(fp)) continue
    if (isFolderPageSourcePath(fp)) {
      for (const folder of folderPageAncestors(folderPageSourceSlug(fp))) {
        navigableFolders.add(folder)
      }
    }
    const fileSlug = slugForDirectoryEntry(fp)
    const ext = path.extname(fp)
    const base = path.basename(fp, ext)
    const md = mdBySlug.get(fileSlug)
    fullTrie.add({
      slug: fileSlug,
      title: md?.frontmatter?.title ?? pdfTitles.get(fileSlug) ?? base,
      filePath: fp,
    })
  }

  return {
    allFiles,
    allPaths: ctx.allFiles,
    aliases,
    descendantsByFolder,
    firstDescendantWithBasename: (folder, base) =>
      firstDescendantByFolderBase.get(folderBaseKey(folder, base)),
    firstDescendantByFolder,
    fullTrie,
    immediateMarkdownByFolder,
    mdBySlug,
    navigableFolders,
    pdfTitles,
  }
}
