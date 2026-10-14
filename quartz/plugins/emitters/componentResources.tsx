import type { QuartzEmitterPlugin } from '../../types/plugin'
import type { FilePath } from '../../util/path'
import { writeAssetManifest } from './component-resources/asset-writer'
import { writeEmojiAssets } from './component-resources/emoji-assets'
import { writeNotebookRuntimeAssets } from './component-resources/notebook-assets'
import { currentComponentResources } from './component-resources/resource-set'
import {
  resolveComponentResourceAssets,
  writePageScripts,
  writeStaticJsResourceBundles,
} from './component-resources/script-assets'
import { writeSharedTrailAsset } from './component-resources/shared-trail-assets'
import { externalResources, writeSiteManifest } from './component-resources/site-manifest'
import {
  writeComponentStyles,
  writeFontAssets,
  writeIndexStylesheet,
  writeStaticCssResourceBundles,
} from './component-resources/style-assets'
import {
  writeCollaborativeCommentsAssets,
  writeGenericWorkerAssets,
  writeSemanticWorkerAssets,
  writeSpeechRuntimeAssets,
} from './component-resources/worker-assets'
import { writeXsltPolyfillAsset } from './component-resources/xslt-polyfill-assets'

const name = 'ComponentResources'

async function* yieldFiles(files: readonly FilePath[]): AsyncGenerator<FilePath> {
  for (const file of files) {
    yield file
  }
}

export const ComponentResources: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, _content, resources) {
      const componentResources = await currentComponentResources(ctx)
      const [
        notebookRuntimeFiles,
        collaborativeCommentsFiles,
        semanticWorkerFiles,
        speechRuntimeFiles,
        emojiFiles,
        xsltPolyfillFile,
        sharedTrailFile,
      ] = await Promise.all([
        writeNotebookRuntimeAssets(ctx),
        writeCollaborativeCommentsAssets(ctx),
        writeSemanticWorkerAssets(ctx),
        writeSpeechRuntimeAssets(ctx),
        writeEmojiAssets(ctx),
        writeXsltPolyfillAsset(ctx),
        writeSharedTrailAsset(ctx),
      ])
      resolveComponentResourceAssets(ctx, componentResources)

      const fontAssets = await writeFontAssets(ctx)
      yield* yieldFiles(fontAssets.files)
      yield* writeComponentStyles(ctx, componentResources)
      yield await writeIndexStylesheet(ctx, componentResources, fontAssets.googleFontsStyleSheet)
      yield* writeStaticCssResourceBundles(ctx, resources)
      yield* writeStaticJsResourceBundles(ctx, resources)
      yield* yieldFiles(await writePageScripts(ctx, componentResources))
      yield await writeSiteManifest(ctx)
      yield* writeGenericWorkerAssets(ctx)
      yield* yieldFiles(notebookRuntimeFiles)
      yield* yieldFiles(collaborativeCommentsFiles)
      yield* yieldFiles(semanticWorkerFiles)
      yield* yieldFiles(speechRuntimeFiles)
      yield* yieldFiles(emojiFiles)
      yield xsltPolyfillFile
      yield sharedTrailFile
      yield writeAssetManifest(ctx)
    },
    // Source edits regenerate resources through buildSourceResources; content edits reuse them.
    async partialEmit() {
      return []
    },
    externalResources,
  }
}
