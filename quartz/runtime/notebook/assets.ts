export const notebookRuntimeClientAsset = 'notebook-runtime.client.js'
export const notebookRuntimeWorkerAsset = 'notebook-runtime.worker.js'
export const notebookRuntimeJavascriptWorkerAsset = 'notebook-runtime.javascript.worker.js'
export const notebookPyrightWorkerManifestAsset = 'notebook-pyright/worker/manifest.json'
export const notebookPyrightTypeshedManifestAsset = 'notebook-pyright/typeshed/manifest.json'
export const notebookPyrightPackageStubsManifestAsset =
  'notebook-pyright/site-packages/manifest.json'

export type NotebookRuntimeAssets = {
  readonly workerUrl: string
  readonly javascriptWorkerUrl: string
  readonly pyrightWorkerManifestUrl: string
  readonly pyrightTypeshedManifestUrl: string
  readonly pyrightPackageStubsManifestUrl: string
}

export type NotebookRuntimeAssetConfig = Partial<NotebookRuntimeAssets>

let configuredAssets: NotebookRuntimeAssetConfig = {}

export function configureNotebookRuntimeAssets(assets: NotebookRuntimeAssetConfig): void {
  configuredAssets = { ...configuredAssets, ...assets }
}

export function notebookRuntimeAssetUrl(
  key: keyof NotebookRuntimeAssets,
  fallback: string,
  baseUrl: string,
): string {
  const configured = configuredAssets[key]
  return configured ? configured : new URL(fallback, baseUrl).href
}
