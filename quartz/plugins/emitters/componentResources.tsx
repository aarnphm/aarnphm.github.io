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
  writeLexicalWorkerAsset,
  writeSemanticWorkerAssets,
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
      const [
        pageAndWorkerScripts,
        fontAndIndexFiles,
        componentStyleFiles,
        staticCssBundleFiles,
        staticJsBundleFiles,
        siteManifestFile,
        genericWorkerFiles,
        speechRuntimeFiles,
        emojiFiles,
        xsltPolyfillFile,
        sharedTrailFiles,
      ] = await Promise.all([
        (async () => {
          const [
            notebookRuntimeFiles,
            collaborativeCommentsFiles,
            semanticWorkerFiles,
            lexicalWorkerFile,
          ] = await Promise.all([
            writeNotebookRuntimeAssets(ctx),
            writeCollaborativeCommentsAssets(ctx),
            writeSemanticWorkerAssets(ctx),
            writeLexicalWorkerAsset(ctx),
          ])
          resolveComponentResourceAssets(ctx, componentResources)
          const pageScriptFiles = await writePageScripts(ctx, componentResources)
          return {
            pageScriptFiles,
            notebookRuntimeFiles,
            collaborativeCommentsFiles,
            semanticWorkerFiles,
            lexicalWorkerFile,
          }
        })(),
        (async () => {
          const fontAssets = await writeFontAssets(ctx)
          const indexStylesheetFile = await writeIndexStylesheet(
            ctx,
            componentResources,
            fontAssets.googleFontsStyleSheet,
          )
          return { fontFiles: fontAssets.files, indexStylesheetFile }
        })(),
        collectFiles(writeComponentStyles(ctx, componentResources)),
        collectFiles(writeStaticCssResourceBundles(ctx, resources)),
        collectFiles(writeStaticJsResourceBundles(ctx, resources)),
        writeSiteManifest(ctx),
        collectFiles(writeGenericWorkerAssets(ctx)),
        writeSpeechRuntimeAssets(ctx),
        writeEmojiAssets(ctx),
        writeXsltPolyfillAsset(ctx),
        writeSharedTrailAsset(ctx),
      ])

      yield* fontAndIndexFiles.fontFiles
      yield* componentStyleFiles
      yield fontAndIndexFiles.indexStylesheetFile
      yield* staticCssBundleFiles
      yield* staticJsBundleFiles
      yield* pageAndWorkerScripts.pageScriptFiles
      yield siteManifestFile
      yield* genericWorkerFiles
      yield* pageAndWorkerScripts.notebookRuntimeFiles
      yield* pageAndWorkerScripts.collaborativeCommentsFiles
      yield* pageAndWorkerScripts.semanticWorkerFiles
      yield pageAndWorkerScripts.lexicalWorkerFile
      yield* speechRuntimeFiles
      yield* emojiFiles
      yield xsltPolyfillFile
      yield* sharedTrailFiles
      yield await writeAssetManifest(ctx)
    },
    // Source edits regenerate resources through buildSourceResources; content edits reuse them.
    async partialEmit() {
      return []
    },
    externalResources,
  }
}
