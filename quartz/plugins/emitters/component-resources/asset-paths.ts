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
export const notebookRuntimePyrightPyodideLockUrl =
  'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide-lock.json'
export const quartzBaseStylesheetEntry = 'quartz/styles/base.scss'
export const quartzCustomStylesheetEntry = 'quartz/styles/custom.scss'
export const basedpyrightSourcePackageName = 'basedpyright-source'
export const basedpyrightInternalSourcePackageName = 'basedpyright-internal-source'
export const notebookPyrightWorkerOutputName = 'notebook-pyright-worker'
export const semanticWorkerEntry = 'quartz/workers/semantic.worker.ts'
export const semanticWorkerPath = `${staticScriptsDir}/semantic.worker.js`
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

export const semanticWorkerAssetEntries = new Set([semanticWorkerEntry, 'package.json'])
