import path from 'path'
import type { FrontmatterLink } from '../plugins/transformers/frontmatter'
import type { AssetManifest } from './asset-manifest'
import type { OutputAssetClaim, OutputAssetManifest } from './output-assets'
import type { StaticResources } from './resources'
import { QuartzConfig } from '../cfg'
import { ProcessedContent, QuartzPluginData } from '../plugins/vfile'
import { FileTrieNode } from './fileTrie'
import { FilePath, FullSlug, SimpleSlug, splitAnchor, stripSlashes } from './path'

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
  slowBuildThreshold?: number
  allBuildSpans?: boolean
}

export type BuildTimeTrieData = QuartzPluginData & { slug: string; title: string; filePath: string }

export type RenderData = {
  source: QuartzPluginData[]
  bySlug: Map<FullSlug, QuartzPluginData>
  baseFiles: QuartzPluginData[]
  backlinksBySlug: Map<SimpleSlug, QuartzPluginData[]>
  recommendationPool: QuartzPluginData[]
  frontmatterLinksByKey: Map<string, Map<FullSlug, FrontmatterLink[]>>
}

export interface BuildCtx {
  buildId: string
  argv: Argv
  cfg: QuartzConfig
  allSlugs: FullSlug[]
  allFiles: FilePath[]
  trie?: FileTrieNode<BuildTimeTrieData>
  incremental: boolean
  gitCommitSha?: string
  assetManifest?: AssetManifest
  extractedStaticResources?: Map<string, string>
  renderData?: RenderData
  pageResourceCacheBuildId?: string
  pageResourceCache?: Map<string, StaticResources>
  cleanOutput?: boolean
  outputAssetManifest?: OutputAssetManifest
  outputAssetClaims?: OutputAssetClaim[]
  outputAssetPreserved?: Set<FilePath>
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

export function renderDataFor(ctx: BuildCtx, allFiles: QuartzPluginData[]): RenderData {
  const cached = ctx.renderData
  if (cached?.source === allFiles) return cached

  const bySlug = new Map<FullSlug, QuartzPluginData>()
  const baseFiles: QuartzPluginData[] = []
  const backlinksBySlug = new Map<SimpleSlug, QuartzPluginData[]>()
  const recommendationPool: QuartzPluginData[] = []
  const frontmatterLinksByKey = new Map<string, Map<FullSlug, FrontmatterLink[]>>()

  for (const file of allFiles) {
    const slug = file.slug as FullSlug | undefined
    if (!slug) continue

    bySlug.set(slug, file)
    if (file.bases) baseFiles.push(file)
    if (!slug.includes('university')) recommendationPool.push(file)

    for (const link of file.links ?? []) {
      const bucket = backlinksBySlug.get(link)
      if (bucket) {
        bucket.push(file)
      } else {
        backlinksBySlug.set(link, [file])
      }
    }

    const frontmatterLinks = file.frontmatterLinks
    if (!frontmatterLinks) continue
    for (const [key, links] of Object.entries(frontmatterLinks)) {
      let byKey = frontmatterLinksByKey.get(key)
      if (!byKey) {
        byKey = new Map()
        frontmatterLinksByKey.set(key, byKey)
      }
      byKey.set(slug, links)
    }
  }
  baseFiles.sort((a, b) => (b.slug?.length ?? 0) - (a.slug?.length ?? 0))

  const renderData = {
    source: allFiles,
    bySlug,
    baseFiles,
    backlinksBySlug,
    recommendationPool,
    frontmatterLinksByKey,
  }
  ctx.renderData = renderData
  return renderData
}

export type WorkerSerializableBuildCtx = Omit<BuildCtx, 'cfg' | 'trie' | 'renderData'>

const contentDataCache = new WeakMap<ProcessedContent[], QuartzPluginData[]>()

export function contentDataFor(content: ProcessedContent[]): QuartzPluginData[] {
  const cached = contentDataCache.get(content)
  if (cached) return cached
  const data = content.map(([, file]) => file.data)
  contentDataCache.set(content, data)
  return data
}
