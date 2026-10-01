import { build as bundle, type Plugin } from 'esbuild'
import path from 'path'
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
  staticScriptsDir,
} from './asset-paths'
import {
  assetSlugForContent,
  staticScriptAssetReference,
  writeAssetBundleOutput,
} from './asset-writer'

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

const inlineScriptNamespace = 'inline-script'
const inlineScriptSpecifier = (index: number) => `${inlineScriptNamespace}:${index}`

// Inline scripts keep npm imports bare (inline-script-bundler.ts), so every script file is bundled
// here and resolves them from the project root. Each script stays its own module scope.
function inlineScriptModules(scripts: readonly string[]): Plugin {
  return {
    name: 'inline-script-modules',
    setup(build) {
      build.onResolve({ filter: /^inline-script:\d+$/ }, args => ({
        path: args.path,
        namespace: inlineScriptNamespace,
      }))
      build.onLoad({ filter: /.*/, namespace: inlineScriptNamespace }, args => ({
        contents: scripts[Number(args.path.slice(inlineScriptNamespace.length + 1))],
        loader: 'js',
        resolveDir: path.resolve('.'),
      }))
    },
  }
}

export async function joinScripts(scripts: string[]): Promise<string> {
  const result = await bundle({
    stdin: {
      contents: scripts.map((_, index) => `import "${inlineScriptSpecifier(index)}"`).join('\n'),
      resolveDir: path.resolve('.'),
    },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'iife',
    write: false,
    plugins: [inlineScriptModules(scripts)],
  })
  return result.outputFiles[0].text
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
  // A repeated script used to hash to the same file, so only its first import evaluated it.
  const sources = [...new Set(scripts)]
  const outdir = path.join(ctx.argv.output, staticScriptsDir)
  // One build for every chunk: packages imported by several scripts move into shared
  // `chunks/` files instead of being copied into each script.
  const result = await bundle({
    entryPoints: Object.fromEntries(
      sources.map((_, index) => [`script-${index}`, inlineScriptSpecifier(index)]),
    ),
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    metafile: true,
    write: false,
    plugins: [inlineScriptModules(sources)],
  })

  const outputs = new Map(result.outputFiles.map(output => [output.path, output]))
  const entryOutputs = sources.map((_, index) => {
    const output = outputs.get(path.resolve(outdir, `script-${index}.js`))
    if (!output) throw new Error(`esbuild emitted no output for page script ${index}`)
    return output
  })
  const entryPaths = new Set(entryOutputs.map(output => output.path))
  const [entries, chunkFiles] = await Promise.all([
    Promise.all(
      entryOutputs.map(async ({ text: content }) => {
        const slug = contentHashSlug('static/scripts/script', content)
        return {
          filename: assetPath(slug, '.js'),
          file: await write({ ctx, slug, ext: '.js', content }),
        }
      }),
    ),
    Promise.all(
      result.outputFiles
        .filter(output => !entryPaths.has(output.path))
        .map(output => writeAssetBundleOutput(ctx, output)),
    ),
  ])

  // Shared chunks reached through static imports load on every page; lazy `import()` chunks stay
  // off the preload list.
  const sharedChunks = new Set<string>()
  const pending = entryOutputs.map(output => path.relative(path.resolve('.'), output.path))
  while (pending.length > 0) {
    for (const imported of result.metafile.outputs[pending.pop()!]?.imports ?? []) {
      if (imported.kind !== 'import-statement' || sharedChunks.has(imported.path)) continue
      sharedChunks.add(imported.path)
      pending.push(imported.path)
    }
  }
  const preloads = [...sharedChunks].map(chunk =>
    path.relative(ctx.argv.output, chunk).split(path.sep).join('/'),
  )

  // modulepreload fetches every chunk and its shared imports in parallel without evaluating them.
  // The awaited imports then evaluate chunks in source order, so spa.inline.ts (last) fires the
  // initial `nav` after every other chunk has attached its listeners; `Promise.all` over `import()`
  // evaluates in arrival order.
  const chunks = JSON.stringify(entries.map(({ filename }) => `./${filename}`))
  const shared = JSON.stringify(preloads.map(chunk => `./${chunk}`))
  const postscript = `const chunks = ${chunks};
for (const src of [...${shared}, ...chunks]) {
  const link = document.createElement("link");
  link.rel = "modulepreload";
  link.href = new URL(src, import.meta.url).href;
  document.head.append(link);
}
for (const src of chunks) await import(src);
`
  return { postscript, files: [...entries.map(({ file }) => file), ...chunkFiles] }
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
