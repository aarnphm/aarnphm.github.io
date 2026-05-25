import type { QuartzEmitterPlugin } from '../../types/plugin'
import type { FilePath } from '../../util/path'
import { writeAssetManifest } from './component-resources/asset-writer'
import { classifyResourceChanges } from './component-resources/change-classifier'
import { writeEmojiAssets } from './component-resources/emoji-assets'
import { writeNotebookRuntimeAssets } from './component-resources/notebook-assets'
import { currentComponentResources } from './component-resources/resource-set'
import {
  resolveComponentResourceAssets,
  writePageScripts,
  writeStaticJsResourceBundles,
} from './component-resources/script-assets'
import { externalResources, writeSiteManifest } from './component-resources/site-manifest'
import {
  writeComponentStyles,
  writeFontAssets,
  writeIndexStylesheet,
  writeStaticCssResourceBundles,
} from './component-resources/style-assets'
import {
  handleGenericWorkerChange,
  removeSemanticWorkerAsset,
  writeCollaborativeCommentsAssets,
  writeGenericWorkerAssets,
  writeSemanticWorkerAssets,
} from './component-resources/worker-assets'

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
      const [notebookRuntimeFiles, collaborativeCommentsFiles, semanticWorkerFiles, emojiFiles] =
        await Promise.all([
          writeNotebookRuntimeAssets(ctx),
          writeCollaborativeCommentsAssets(ctx),
          writeSemanticWorkerAssets(ctx),
          writeEmojiAssets(ctx),
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
      yield* yieldFiles(emojiFiles)
      yield writeAssetManifest(ctx)
    },
    async *partialEmit(ctx, _content, resources, changeEvents) {
      const changes = classifyResourceChanges(changeEvents)
      const componentResources = await currentComponentResources(ctx)
      yield* writeComponentStyles(ctx, componentResources)
      yield* writeStaticCssResourceBundles(ctx, resources)
      yield* writeStaticJsResourceBundles(ctx, resources)

      if (changes.indexStylesheet) {
        yield await writeIndexStylesheet(ctx, componentResources)
      }

      if (changes.notebookRuntime) {
        yield* yieldFiles(await writeNotebookRuntimeAssets(ctx))
      }

      if (changes.notebookRuntimePageScript) {
        resolveComponentResourceAssets(ctx, componentResources)
        yield* yieldFiles(await writePageScripts(ctx, componentResources))
      }

      if (changes.collaborativeComments) {
        yield* yieldFiles(await writeCollaborativeCommentsAssets(ctx))
      }

      if (changes.semanticWorkerDeleted) {
        await removeSemanticWorkerAsset(ctx)
      } else if (changes.semanticWorker) {
        yield* yieldFiles(await writeSemanticWorkerAssets(ctx))
      }

      if (changes.emoji) {
        yield* yieldFiles(await writeEmojiAssets(ctx))
      }

      for (const changeEvent of changes.genericWorkerChanges) {
        const file = await handleGenericWorkerChange(ctx, changeEvent)
        if (file) yield file
      }

      yield writeAssetManifest(ctx)
    },
    externalResources,
  }
}
