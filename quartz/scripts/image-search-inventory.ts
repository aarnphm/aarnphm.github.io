import { globby } from 'globby'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { glob } from '../util/glob'
import { isCourseMaterialPath } from '../util/image-search-scope'
import { isFilePath, slugifyFilePath } from '../util/path'
import { isSearchExcluded } from '../util/search-exclude'
import { isRecord } from '../util/type-guards'
import {
  IMAGE_EXTENSION,
  parseImageReferences,
  type ImageReferenceSource,
} from './image-search/references'

export { isCourseMaterialPath } from '../util/image-search-scope'

export interface ImageOccurrence {
  slug: string
  title: string
  caption?: string
  source: ImageReferenceSource
  public: boolean
  exclusionReasons: string[]
}
export interface ImageInventoryAsset {
  assetId: string
  sourcePath: string
  absolutePath: string
  imageUrl: string
  contentId: string
  bytes: number
  public: boolean
  exclusionReasons: string[]
  occurrences: ImageOccurrence[]
}
export interface ImageInventoryIssue {
  sourcePath: string
  reference?: string
  reason: string
}
export interface ImageSearchInventory {
  version: 1
  assets: ImageInventoryAsset[]
  issues: ImageInventoryIssue[]
  digest: string
}
export type ImageInventoryOptions = {
  contentDir: string
  ignorePatterns?: readonly string[]
  publicationOnly?: boolean
} & (
  | { publicIndex: Readonly<Record<string, unknown>>; publicIndexPath?: string }
  | { publicIndex?: undefined; publicIndexPath: string }
)

function slugFor(sourcePath: string): string {
  const prefixed = `content/${sourcePath}`
  if (!isFilePath(prefixed)) throw new Error(`Invalid source path: ${sourcePath}`)
  return slugifyFilePath(prefixed).slice('content/'.length)
}

function contextExclusions(frontmatter: Record<string, unknown>): string[] {
  const reasons = ['private', 'draft', 'protected', 'password', 'noindex', 'unlisted'].filter(
    key => Boolean(frontmatter[key]) && frontmatter[key] !== 'false',
  )
  if (frontmatter.semantic === false) reasons.push('semantic-false')
  return reasons
}

async function identifyImage(
  asset: ImageInventoryAsset,
  issues: ImageInventoryIssue[],
): Promise<void> {
  let data: Buffer
  try {
    data = await readFile(asset.absolutePath)
  } catch (error) {
    issues.push({
      sourcePath: asset.sourcePath,
      reason: `read-error: ${error instanceof Error ? error.message : String(error)}`,
    })
    asset.exclusionReasons.push('read-error')
    return
  }
  if (data.subarray(0, 120).toString().startsWith('version https://git-lfs.github.com/spec/v1')) {
    asset.exclusionReasons.push('lfs-pointer')
    issues.push({ sourcePath: asset.sourcePath, reason: 'lfs-pointer' })
  }
  asset.contentId = createHash('sha256').update(data).digest('hex')
  asset.bytes = data.length
}

export async function collectImageSearchInventory(
  options: ImageInventoryOptions,
): Promise<ImageSearchInventory> {
  const contentDir = path.resolve(options.contentDir)
  if (!options.publicIndex && !options.publicIndexPath)
    throw new Error('Published metadata is required')
  const index: unknown =
    options.publicIndex ??
    (options.publicIndexPath
      ? JSON.parse(await readFile(options.publicIndexPath, 'utf8'))
      : undefined)
  if (!isRecord(index)) throw new Error('Invalid published content index')
  const ignorePatterns = options.ignorePatterns ?? ['private', 'templates', '.obsidian']
  const sourcePaths = (
    await globby('**/*', {
      cwd: contentDir,
      dot: true,
      gitignore: false,
      followSymbolicLinks: false,
      ignore: ['**/node_modules/**', '**/.git/**', '**/.venv/**', '**/__pycache__/**'],
    })
  ).sort()
  const publicCandidates = new Set<string>(await glob('**', contentDir, [...ignorePatterns]))
  const issues: ImageInventoryIssue[] = []
  const assets: ImageInventoryAsset[] = []
  for (const sourcePath of sourcePaths.filter(file => IMAGE_EXTENSION.test(file))) {
    const exclusionReasons: string[] = []
    if (!publicCandidates.has(sourcePath)) exclusionReasons.push('ignored-path')
    if (isSearchExcluded(sourcePath)) exclusionReasons.push('search-excluded')
    if (isCourseMaterialPath(sourcePath)) exclusionReasons.push('course-material')
    const absolutePath = path.join(contentDir, sourcePath)
    const assetId = slugFor(sourcePath)
    const asset: ImageInventoryAsset = {
      assetId,
      sourcePath,
      absolutePath,
      imageUrl: `/${assetId}`,
      contentId: '',
      bytes: 0,
      public: false,
      exclusionReasons,
      occurrences: [],
    }
    if (!options.publicationOnly) await identifyImage(asset, issues)
    assets.push(asset)
  }
  const byPath = new Map(assets.map(asset => [asset.sourcePath, asset]))
  const bySlug = new Map(assets.map(asset => [asset.assetId, asset]))
  const byBasename = new Map<string, ImageInventoryAsset[]>()
  for (const asset of assets) {
    const basename = path.posix.basename(asset.sourcePath)
    byBasename.set(basename, [...(byBasename.get(basename) ?? []), asset])
  }
  for (const sourcePath of sourcePaths.filter(file => /\.(?:md|base|canvas|ipynb)$/i.test(file))) {
    const slug = slugFor(sourcePath)
    const indexed = index[slug]
    if (
      options.publicationOnly &&
      (!isRecord(indexed) ||
        contextExclusions(indexed).length > 0 ||
        !publicCandidates.has(sourcePath) ||
        isSearchExcluded(sourcePath) ||
        isCourseMaterialPath(sourcePath))
    )
      continue
    let parsed: ReturnType<typeof parseImageReferences>
    try {
      parsed = parseImageReferences(
        sourcePath,
        await readFile(path.join(contentDir, sourcePath), 'utf8'),
        options.publicationOnly
          ? frontmatter => contextExclusions(frontmatter).length === 0
          : undefined,
      )
    } catch (error) {
      issues.push({
        sourcePath,
        reason: `parse-error: ${error instanceof Error ? error.message : String(error)}`,
      })
      continue
    }
    const exclusionReasons = contextExclusions(parsed.frontmatter)
    if (!isRecord(indexed)) exclusionReasons.push('unpublished-note')
    if (isRecord(indexed) && indexed.protected) exclusionReasons.push('protected')
    if (!publicCandidates.has(sourcePath)) exclusionReasons.push('ignored-path')
    if (isSearchExcluded(sourcePath)) exclusionReasons.push('search-excluded')
    if (isCourseMaterialPath(sourcePath)) exclusionReasons.push('course-material')
    const title =
      typeof parsed.frontmatter.title === 'string'
        ? parsed.frontmatter.title
        : isRecord(indexed) && typeof indexed.title === 'string'
          ? indexed.title
          : path.posix.basename(sourcePath, path.posix.extname(sourcePath))
    for (const reference of parsed.references) {
      let target: string
      try {
        target = decodeURIComponent(reference.target.split(/[?#]/)[0]).replace(/\\/g, '/')
      } catch {
        issues.push({ sourcePath, reference: reference.target, reason: 'invalid-reference' })
        continue
      }
      if (!IMAGE_EXTENSION.test(target)) continue
      if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(target)) {
        issues.push({ sourcePath, reference: reference.target, reason: 'remote-reference' })
        continue
      }
      const relative = target.startsWith('.')
        ? path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), target))
        : target.replace(/^\//, '')
      if (relative.startsWith('../') || path.posix.isAbsolute(relative)) {
        issues.push({ sourcePath, reference: reference.target, reason: 'outside-vault' })
        continue
      }
      let asset = byPath.get(relative) ?? bySlug.get(relative)
      if (!asset && !target.startsWith('/') && !target.startsWith('.')) {
        asset = byPath.get(path.posix.join(path.posix.dirname(sourcePath), target))
        if (!asset && !target.includes('/')) {
          const matches = byBasename.get(target) ?? []
          if (matches.length > 1) {
            issues.push({ sourcePath, reference: reference.target, reason: 'ambiguous-reference' })
            continue
          }
          asset = matches[0]
        }
      }
      if (!asset) {
        issues.push({ sourcePath, reference: reference.target, reason: 'missing-reference' })
        continue
      }
      asset.occurrences.push({
        slug,
        title,
        caption: reference.caption,
        source: reference.source,
        public: exclusionReasons.length === 0,
        exclusionReasons: [...new Set(exclusionReasons)],
      })
    }
  }
  for (const asset of assets) {
    if (
      asset.occurrences.length &&
      asset.occurrences.every(occurrence =>
        occurrence.exclusionReasons.includes('course-material'),
      ) &&
      !asset.exclusionReasons.includes('course-material')
    )
      asset.exclusionReasons.push('course-material')
    if (!asset.occurrences.length) asset.exclusionReasons.push('orphan')
    else if (!asset.occurrences.some(occurrence => occurrence.public))
      asset.exclusionReasons.push('no-public-occurrence')
    if (options.publicationOnly && asset.exclusionReasons.length === 0)
      await identifyImage(asset, issues)
    asset.public = asset.exclusionReasons.length === 0
  }
  const resultAssets = options.publicationOnly ? assets.filter(asset => asset.public) : assets
  const digest = createHash('sha256')
    .update(JSON.stringify(resultAssets.map(asset => ({ ...asset, absolutePath: undefined }))))
    .digest('hex')
  return { version: 1, assets: resultAssets, issues, digest }
}
