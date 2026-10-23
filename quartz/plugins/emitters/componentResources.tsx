import type { QuartzEmitterPlugin } from '../../types/plugin'
import type { FilePath } from '../../util/path'
import { allSettledOrThrow } from '../../util/staged-output'
import { writeAssetManifest } from './component-resources/asset-writer'
import { writeEmojiAssets } from './component-resources/emoji-assets'
import { currentComponentResources } from './component-resources/resource-set'
import { writePageScripts, writeStaticJsResourceBundles } from './component-resources/script-assets'
import { writeSharedTrailAsset } from './component-resources/shared-trail-assets'
import { externalResources, writeSiteManifest } from './component-resources/site-manifest'
import {
  writeComponentStyles,
  writeFontAssets,
  writeIndexStylesheet,
  writeStaticCssResourceBundles,
} from './component-resources/style-assets'
import {
  writeGenericWorkerAssets,
  writeSpeechRuntimeAssets,
} from './component-resources/worker-assets'
import { writeXsltPolyfillAsset } from './component-resources/xslt-polyfill-assets'

const name = 'ComponentResources'

async function collectFiles(files: AsyncIterable<FilePath>): Promise<FilePath[]> {
  const collected: FilePath[] = []
  for await (const file of files) {
    collected.push(file)
  }
  return collected
}

export const ComponentResources: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, _content, resources) {
      const componentResources = await currentComponentResources(ctx)
      const resourceFiles = await allSettledOrThrow([
        writePageScripts(ctx, componentResources),
        (async () => {
          const fontAssets = await writeFontAssets(ctx)
          const indexStylesheetFile = await writeIndexStylesheet(
            ctx,
            componentResources,
            fontAssets.googleFontsStyleSheet,
          )
          return [...fontAssets.files, indexStylesheetFile]
        })(),
        collectFiles(writeComponentStyles(ctx, componentResources)),
        collectFiles(writeStaticCssResourceBundles(ctx, resources)),
        collectFiles(writeStaticJsResourceBundles(ctx, resources)),
        writeSiteManifest(ctx).then(file => [file]),
        collectFiles(writeGenericWorkerAssets(ctx)),
        writeSpeechRuntimeAssets(ctx),
        writeEmojiAssets(ctx),
        writeXsltPolyfillAsset(ctx).then(file => [file]),
        writeSharedTrailAsset(ctx),
      ])

      for (const files of resourceFiles) yield* files
      yield await writeAssetManifest(ctx)
    },
    // Source edits regenerate resources through buildSourceResources; content edits reuse them.
    async partialEmit() {
      return []
    },
    externalResources,
  }
}
