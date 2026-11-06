import type { QuartzPluginData } from '../../plugins/vfile'
import type { BuildCtx } from '../ctx'
import type { ReferenceCatalog, ReferenceCatalogInput } from './catalog-types'
import { getFileExtension } from '../path'
import { createReferenceCatalog } from './catalog'

const contexts = new WeakMap<
  BuildCtx,
  {
    files: BuildCtx['allFiles']
    inputs: BuildCtx['referenceCatalogInputs']
    catalog: ReferenceCatalog
  }
>()
const fileCatalogs = new WeakMap<readonly QuartzPluginData[], ReferenceCatalog>()

export function referenceSourcePath(file: QuartzPluginData): string {
  if (file.relativePath) return file.relativePath
  const extension = getFileExtension(file.filePath ?? '') ?? '.md'
  const slug = file.slug ?? ''
  return getFileExtension(slug) ? slug : `${slug}${extension}`
}

function fromFile(file: QuartzPluginData): ReferenceCatalogInput {
  if (file.referenceCatalogInput) return file.referenceCatalogInput
  return {
    path: referenceSourcePath(file),
    slug: file.slug,
    aliases: file.frontmatter?.aliases,
    headings: file.toc?.map(heading => ({
      text: heading.text,
      anchor: heading.slug,
      depth: heading.depth,
    })),
    blocks: file.blocks ? Object.keys(file.blocks) : undefined,
    baseViews: file.basesConfig?.views.map(view => ({ name: view.name })),
    ...(file.frontmatter?.pageLayout === 'latent' ? { latent: true } : {}),
  }
}

/** Each build owns a serializable metadata snapshot; processors may outlive that snapshot. */
export function referenceCatalogFor(ctx: BuildCtx): ReferenceCatalog {
  const cached = contexts.get(ctx)
  if (cached?.files === ctx.allFiles && cached.inputs === ctx.referenceCatalogInputs)
    return cached.catalog
  const directory = ctx.argv.directory.replaceAll('\\', '/').replace(/\/$/, '')
  const metadata = new Map(ctx.referenceCatalogInputs?.map(input => [input.path, input]))
  const paths = ctx.allFiles.map(file => {
    const source = file.replaceAll('\\', '/')
    return source.startsWith(`${directory}/`) ? source.slice(directory.length + 1) : source
  })
  const catalog = createReferenceCatalog(
    paths.map(source => metadata.get(source) ?? { path: source }),
  )
  contexts.set(ctx, { files: ctx.allFiles, inputs: ctx.referenceCatalogInputs, catalog })
  return catalog
}

/** Render and Base callers share the same source-identity adapter. */
export function referenceCatalogFromFiles(files: readonly QuartzPluginData[]): ReferenceCatalog {
  const cached = fileCatalogs.get(files)
  if (cached) return cached
  const catalog = createReferenceCatalog(files.map(fromFile))
  fileCatalogs.set(files, catalog)
  return catalog
}

export function updateReferenceCatalog(ctx: BuildCtx, files: readonly QuartzPluginData[]): void {
  const inputs = new Map(ctx.referenceCatalogInputs?.map(input => [input.path, input]))
  for (const file of files) {
    const input = fromFile(file)
    inputs.set(input.path, input)
  }
  ctx.referenceCatalogInputs = [...inputs.values()]
  // Prune deleted identities using the refreshed path catalog.
  ctx.referenceCatalogInputs = referenceCatalogFor(ctx).entries
}

declare module 'vfile' {
  interface DataMap {
    referenceCatalogInput?: ReferenceCatalogInput
  }
}
