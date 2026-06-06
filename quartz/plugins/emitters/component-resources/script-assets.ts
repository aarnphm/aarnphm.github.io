import { transform as transpile } from 'esbuild'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath } from '../../../util/path'
import type { StaticResources } from '../../../util/resources'
import type { ComponentResourceSet } from './resource-set'
import {
  notebookNativeRuntimeManifestAsset,
  notebookPyrightPackageStubsManifestAsset,
  notebookPyrightTypeshedManifestAsset,
  notebookPyrightWorkerManifestAsset,
  notebookRuntimeClientAsset,
  notebookRuntimeJavascriptWorkerAsset,
  notebookRuntimeWorkerAsset,
} from '../../../runtime/notebook/assets'
import {
  assetPath,
  contentHashSlug,
  registerExtractedStaticResource,
  shouldHashAssets,
} from '../../../util/asset-manifest'
import { bundleInlineScript } from '../../../util/inline-script-bundler'
import {
  splitJsBundles,
  staticJsBundleKey,
  staticJsBundleSlug,
} from '../../../util/resource-bundles'
import { write } from '../helpers'
import {
  collaborativeCommentsClientPath,
  notebookNativeRuntimeManifestPath,
  notebookPyrightPackageStubsManifestPath,
  notebookPyrightTypeshedManifestPath,
  notebookPyrightWorkerManifestPath,
  notebookRuntimeClientPath,
  notebookRuntimeJavascriptWorkerPath,
  notebookRuntimeWorkerPath,
  semanticWorkerPath,
} from './asset-paths'
import { assetSlugForContent, staticScriptAssetReference } from './asset-writer'

export type ScriptAssetReplacement = { placeholder: string; logicalPath: string }

export const componentScriptAssetReplacements: ScriptAssetReplacement[] = [
  { placeholder: notebookRuntimeClientAsset, logicalPath: notebookRuntimeClientPath },
  { placeholder: notebookRuntimeWorkerAsset, logicalPath: notebookRuntimeWorkerPath },
  {
    placeholder: notebookRuntimeJavascriptWorkerAsset,
    logicalPath: notebookRuntimeJavascriptWorkerPath,
  },
  {
    placeholder: notebookNativeRuntimeManifestAsset,
    logicalPath: notebookNativeRuntimeManifestPath,
  },
  {
    placeholder: notebookPyrightWorkerManifestAsset,
    logicalPath: notebookPyrightWorkerManifestPath,
  },
  {
    placeholder: notebookPyrightTypeshedManifestAsset,
    logicalPath: notebookPyrightTypeshedManifestPath,
  },
  {
    placeholder: notebookPyrightPackageStubsManifestAsset,
    logicalPath: notebookPyrightPackageStubsManifestPath,
  },
  { placeholder: 'collaborative-comments.client.js', logicalPath: collaborativeCommentsClientPath },
  { placeholder: 'semantic.worker.js', logicalPath: semanticWorkerPath },
]

export async function joinScripts(scripts: string[]): Promise<string> {
  const script = scripts.map(script => `(function () {${script}})();`).join('\n')
  const res = await transpile(script, { minify: true })
  return res.code
}

export function resolveComponentResourceAssets(
  ctx: BuildCtx,
  componentResources: ComponentResourceSet,
  replacements: readonly ScriptAssetReplacement[] = componentScriptAssetReplacements,
): void {
  componentResources.afterDOMLoaded = componentResources.afterDOMLoaded.map(script =>
    replacements.reduce(
      (current, replacement) =>
        current.replaceAll(
          replacement.placeholder,
          staticScriptAssetReference(ctx, replacement.logicalPath),
        ),
      script,
    ),
  )
}

export async function* writeStaticJsResourceBundles(
  ctx: BuildCtx,
  resources: StaticResources,
): AsyncGenerator<FilePath> {
  for (const loadTime of ['beforeDOMReady', 'afterDOMReady'] as const) {
    const leadingInline = await staticJsLeadingInline(loadTime)
    ctx.staticLeadingJs ??= {}
    ctx.staticLeadingJs[loadTime] = leadingInline
    let index = 0
    for (const part of splitJsBundles(resources.js, loadTime, leadingInline)) {
      if (part.type !== 'bundle') continue
      const content = await joinScripts(part.scripts)
      const baseSlug = staticJsBundleSlug(part.loadTime)
      const devSlug = index === 0 ? baseSlug : `${baseSlug}-${index}`
      const slug = shouldHashAssets(ctx) ? contentHashSlug(baseSlug, content) : devSlug
      index += 1
      registerExtractedStaticResource(
        ctx,
        staticJsBundleKey(part.loadTime, part.scripts),
        assetPath(slug, '.js'),
      )
      yield write({ ctx, slug, ext: '.js', content })
    }
  }
}

async function staticJsLeadingInline(loadTime: 'beforeDOMReady' | 'afterDOMReady') {
  if (loadTime === 'beforeDOMReady') return []
  return Promise.all([
    bundleInlineScript('quartz/components/scripts/pdf.inline.ts'),
    bundleInlineScript('quartz/components/scripts/transclude.inline.ts'),
    bundleInlineScript('quartz/components/scripts/collapse-header.inline.ts'),
  ])
}

async function writeAfterDomLoadedScripts(
  ctx: BuildCtx,
  scripts: readonly string[],
): Promise<{ postscript: string; files: FilePath[] }> {
  const entries = await Promise.all(
    scripts.map(async script => {
      const content = await joinScripts([script])
      const slug = contentHashSlug('static/scripts/script', content)
      return {
        filename: assetPath(slug, '.js'),
        file: await write({ ctx, slug, ext: '.js', content }),
      }
    }),
  )

  const postscript = entries.map(({ filename }) => `await import("./${filename}");`).join('\n')
  return { postscript, files: entries.map(({ file }) => file) }
}

export async function writePageScripts(
  ctx: BuildCtx,
  componentResources: ComponentResourceSet,
): Promise<FilePath[]> {
  const [prescript, postscriptResult] = await Promise.all([
    joinScripts(componentResources.beforeDOMLoaded),
    writeAfterDomLoadedScripts(ctx, componentResources.afterDOMLoaded),
  ])
  const { postscript, files } = postscriptResult
  const prescriptFile = await write({
    ctx,
    slug: assetSlugForContent(ctx, 'prescript', '.js', prescript),
    ext: '.js',
    content: prescript,
  })
  const postscriptFile = await write({
    ctx,
    slug: assetSlugForContent(ctx, 'postscript', '.js', postscript),
    ext: '.js',
    content: postscript,
  })
  return [prescriptFile, postscriptFile, ...files]
}
