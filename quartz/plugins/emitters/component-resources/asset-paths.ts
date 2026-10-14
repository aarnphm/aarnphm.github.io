import {
  notebookPyrightPackageStubsManifestAsset,
  notebookPyrightTypeshedManifestAsset,
  notebookPyrightWorkerManifestAsset,
  notebookNativeRuntimeManifestAsset,
  notebookRuntimeClientAsset,
  notebookRuntimeJavascriptWorkerAsset,
  notebookRuntimeWorkerAsset,
} from '../../../runtime/notebook/assets'

export const staticScriptsDir = 'static/scripts'
export const componentResourcesEntry = 'quartz/plugins/emitters/componentResources.tsx'
export const collaborativeCommentsClientEntry =
  'quartz/components/scripts/collaborative-comments.client.ts'
export const notebookRuntimeInlineEntry = 'quartz/components/scripts/notebook-runtime.inline.ts'
export const notebookRuntimeClientEntry = 'quartz/runtime/notebook/client.ts'
export const notebookRuntimeLspEntry = 'quartz/runtime/lsp/pyright.ts'
export const notebookRuntimeWorkerEntry = 'quartz/runtime/python/pyodide-worker.js'
export const notebookRuntimeJavascriptWorkerEntry = 'quartz/runtime/javascript/browser-worker.ts'
export const notebookRuntimeBootstrapEntry = 'quartz/runtime/python/pyodide-bridge.py'
export const notebookRuntimeMlBridgeEntry = 'quartz/runtime/python/ml-bridge.js'
export const notebookRuntimePyrightWorkerEntry = 'quartz/util/pyright-browser-worker.js'
export const notebookRuntimePyrightWorkerGlobalsEntry = 'quartz/util/pyright-worker-globals.js'
export const notebookRuntimePyrightTypeshedDir =
  'node_modules/basedpyright/dist/typeshed-fallback/stdlib'
export const notebookRuntimePyrightPackageStubsDir = 'quartz/util/pyright-stubs/site-packages'
export const notebookRuntimePyrightGenericStubPath = 'quartz/util/pyright-stubs/generic-module.pyi'
export const notebookRuntimePyrightSitePackagesPath = '/site-packages'
export const notebookRuntimePyrightPyodideLockPath = 'quartz/runtime/lsp/pyodide-lock-v0.29.4.json'
export const notebookNativeRuntimePacksDir = 'quartz/runtime/native/packs'
export const notebookNativeRuntimeManifestSourcePath = `${notebookNativeRuntimePacksDir}/manifest.json`
export const quartzBaseStylesheetEntry = 'quartz/styles/base.scss'
export const quartzCustomStylesheetEntry = 'quartz/styles/custom.scss'
export const basedpyrightSourcePackageName = 'basedpyright-source'
export const basedpyrightInternalSourcePackageName = 'basedpyright-internal-source'
export const notebookPyrightWorkerOutputName = 'notebook-pyright-worker'
export const semanticWorkerEntry = 'quartz/workers/semantic.worker.ts'
export const semanticWorkerPath = `${staticScriptsDir}/semantic.worker.js`
export const xsltPolyfillPath = `${staticScriptsDir}/xslt-polyfill.js`
export const emojiAssetSourceDir = 'quartz/util/emojimap'
export const notebookRuntimeClientPath = `${staticScriptsDir}/${notebookRuntimeClientAsset}`
export const notebookRuntimeWorkerPath = `${staticScriptsDir}/${notebookRuntimeWorkerAsset}`
export const notebookRuntimeJavascriptWorkerPath = `${staticScriptsDir}/${notebookRuntimeJavascriptWorkerAsset}`
export const notebookNativeRuntimeManifestPath = `${staticScriptsDir}/${notebookNativeRuntimeManifestAsset}`
export const collaborativeCommentsClientPath = `${staticScriptsDir}/collaborative-comments.client.js`
export const notebookPyrightWorkerManifestPath = `${staticScriptsDir}/${notebookPyrightWorkerManifestAsset}`
export const notebookPyrightTypeshedManifestPath = `${staticScriptsDir}/${notebookPyrightTypeshedManifestAsset}`
export const notebookPyrightPackageStubsManifestPath = `${staticScriptsDir}/${notebookPyrightPackageStubsManifestAsset}`
export const notebookPyrightTypeshedBaseDir = `${staticScriptsDir}/notebook-pyright/typeshed`
export const notebookPyrightPackageStubsBaseDir = `${staticScriptsDir}/notebook-pyright/site-packages`

export const notebookRuntimeAssetEntries = new Set([
  componentResourcesEntry,
  notebookRuntimeClientEntry,
  notebookRuntimeLspEntry,
  notebookRuntimeWorkerEntry,
  notebookRuntimeJavascriptWorkerEntry,
  notebookRuntimeBootstrapEntry,
  notebookRuntimeMlBridgeEntry,
  notebookRuntimePyrightWorkerEntry,
  notebookRuntimePyrightWorkerGlobalsEntry,
  notebookRuntimePyrightPyodideLockPath,
  'quartz/runtime/lsp/pyright-assets.ts',
  'quartz/util/type-guards.ts',
  'quartz/runtime/notebook/assets.ts',
  'quartz/runtime/editor/code-editor.ts',
])

export const notebookRuntimeAssetPrefixes = [
  'quartz/plugins/emitters/component-resources/',
  'quartz/util/notebook/',
  'quartz/runtime/',
  'quartz/util/pyright-stubs/',
]

export const collaborativeCommentsAssetPrefixes = [
  'quartz/components/multiplayer/',
  'quartz/functional/',
]

export const collaborativeCommentsAssetEntries = new Set([
  collaborativeCommentsClientEntry,
  'quartz/util/type-guards.ts',
  'quartz/components/scripts/markdown-editor.ts',
  'quartz/components/scripts/search-index.ts',
])

export const semanticWorkerAssetEntries = new Set([
  semanticWorkerEntry,
  'quartz/util/semantic-search.ts',
  'package.json',
])

export const paceWorkerEntry = 'quartz/workers/pace.worker.ts'
export const paceWorkerAssetEntries = new Set([
  paceWorkerEntry,
  'quartz/util/pace-features.ts',
  'quartz/util/pace-model.ts',
  'quartz/util/type-guards.ts',
])

// CLI sync/dev helpers and Cloudflare Worker-only helpers under `quartz/util/` that no browser
// script or server build imports.
export const nonClientUtilEntries = new Set([
  'quartz/util/curius-feed.ts',
  'quartz/util/dev-manager-state.ts',
  'quartz/util/env-file.ts',
  'quartz/util/garmin-connect.ts',
  'quartz/util/garmin-session.ts',
  'quartz/util/garmin-tcx-fit.ts',
  'quartz/util/garmin-title-sync.ts',
  'quartz/util/garmin-title-update.ts',
  'quartz/util/hast-query.ts',
  'quartz/util/lesswrong-preview.ts',
  'quartz/util/not-found.ts',
  'quartz/util/process-tree.ts',
  'quartz/util/sep-preview.ts',
  'quartz/util/strava-sync-window.ts',
  'quartz/util/strava-tcx.ts',
  'quartz/util/sync-refresh-window.ts',
  'quartz/util/trainingpeaks-api.ts',
  'quartz/util/trainingpeaks-title-sync.ts',
  'quartz/util/triathlon-cache.ts',
  'quartz/util/wahoo-cloud.ts',
  'quartz/util/weather-kit.ts',
  'quartz/util/weather-route-hours.ts',
])

export const xsltPolyfillAssetEntries = new Set([
  componentResourcesEntry,
  'package.json',
  'pnpm-lock.yaml',
  'quartz/plugins/emitters/component-resources/asset-paths.ts',
  'quartz/plugins/emitters/component-resources/xslt-polyfill-assets.ts',
])
