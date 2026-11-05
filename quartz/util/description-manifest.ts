import { readFile } from 'node:fs/promises'
import path from 'node:path'
import type { BuildCtx } from './ctx'
import type { ReferenceCatalog } from './references/catalog'
import { parseDescriptionManifest } from './description'
import { ReloadableWeakMap } from './module-caches'
import { referenceCatalogFor } from './references/context'

export const DESCRIPTION_MANIFEST_FILE = 'description.json'

const manifestSnapshots = new ReloadableWeakMap<
  BuildCtx,
  { buildId: string; catalog: ReferenceCatalog; map: Promise<ReadonlyMap<string, string>> }
>()

export function descriptionManifestPath(ctx: BuildCtx): string {
  return path.posix.join(ctx.argv.directory, DESCRIPTION_MANIFEST_FILE)
}

export function loadDescriptionManifest(ctx: BuildCtx): Promise<ReadonlyMap<string, string>> {
  const catalog = referenceCatalogFor(ctx)
  let snapshot = manifestSnapshots.get(ctx)
  if (!snapshot || snapshot.buildId !== ctx.buildId || snapshot.catalog !== catalog) {
    const file = descriptionManifestPath(ctx)
    snapshot = {
      buildId: ctx.buildId,
      catalog,
      map: readFile(file, 'utf8').then(
        raw => {
          try {
            return parseDescriptionManifest(JSON.parse(raw), {
              catalog,
              origin: 'description.json',
            })
          } catch {
            return new Map<string, string>()
          }
        },
        () => new Map<string, string>(),
      ),
    }
    manifestSnapshots.set(ctx, snapshot)
  }
  return snapshot.map
}
