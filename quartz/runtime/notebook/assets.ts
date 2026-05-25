export type NotebookRuntimeAssets = {
  readonly workerUrl: string
  readonly javascriptWorkerUrl: string
  readonly pyrightWorkerManifestUrl: string
  readonly pyrightTypeshedManifestUrl: string
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
