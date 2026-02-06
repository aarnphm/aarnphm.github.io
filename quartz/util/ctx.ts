import path from 'path'
import { QuartzConfig } from '../cfg'
import { QuartzPluginData } from '../plugins/vfile'
import { FileTrieNode } from './fileTrie'
import { FilePath, FullSlug, splitAnchor, stripSlashes } from './path'

export interface Argv {
  directory: string
  verbose: boolean
  output: string
  serve: boolean
  watch: boolean
  port: number
  wsPort: number
  force: boolean
  remoteDevHost?: string
  concurrency?: number
}

export type BuildTimeTrieData = QuartzPluginData & { slug: string; title: string; filePath: string }

export interface BuildCtx {
  buildId: string
  argv: Argv
  cfg: QuartzConfig
  allSlugs: FullSlug[]
  allFiles: FilePath[]
  trie?: FileTrieNode<BuildTimeTrieData>
  incremental: boolean
  gitCommitSha?: string
}

export function trieFromAllFiles(allFiles: QuartzPluginData[]): FileTrieNode<BuildTimeTrieData> {
  const trie = new FileTrieNode<BuildTimeTrieData>([])
  allFiles.forEach(file => {
    // Handle PDFs and files with frontmatter
    if (file.slug) {
      const isPdf = file.filePath
        ? path.extname(file.filePath).toLowerCase().includes('pdf')
        : false

      if (isPdf || file.frontmatter) {
        let slug = file.slug
        let title = file.frontmatter?.title

        // Special handling for PDFs
        if (isPdf) {
          // Ensure the slug is properly formatted for PDFs
          const url = new URL(`/${file.slug}`, 'https://base.com')
          const canonicalDest = url.pathname
          const [destCanonical, _] = splitAnchor(canonicalDest)
          slug = decodeURIComponent(stripSlashes(destCanonical, true)) as FullSlug

          // Use filename as title if no frontmatter title
          if (!title) {
            const baseName = path.basename(file.filePath!, '.pdf')
            title = baseName
          }
        }

        trie.add({ ...file, slug, title: title || slug, filePath: file.filePath! })
      }
    }
  })

  return trie
}

export type WorkerSerializableBuildCtx = Omit<BuildCtx, 'cfg' | 'trie'>
