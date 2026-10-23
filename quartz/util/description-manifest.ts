import { readFile } from 'node:fs/promises'
import path from 'node:path'
import type { BuildCtx } from './ctx'
import { parseDescriptionManifest } from './description'
import { ReloadableWeakMap } from './module-caches'

export const DESCRIPTION_MANIFEST_FILE = 'description.json'

const manifestSnapshots = new ReloadableWeakMap<
  BuildCtx,
  { buildId: string; map: Promise<ReadonlyMap<string, string>> }
>()

export function descriptionManifestPath(ctx: BuildCtx): string {
  return path.posix.join(ctx.argv.directory, DESCRIPTION_MANIFEST_FILE)
}

export function loadDescriptionManifest(ctx: BuildCtx): Promise<ReadonlyMap<string, string>> {
  let snapshot = manifestSnapshots.get(ctx)
  if (!snapshot || snapshot.buildId !== ctx.buildId) {
    const file = descriptionManifestPath(ctx)
    snapshot = {
      buildId: ctx.buildId,
      map: readFile(file, 'utf8').then(
        raw => {
          try {
            return parseDescriptionManifest(JSON.parse(raw))
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
