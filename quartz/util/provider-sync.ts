export type ProviderSync = number | { readonly lastSync: number } | null | undefined

export function latestProviderSync(...providers: readonly ProviderSync[]): number {
  let latest = 0
  for (const provider of providers) {
    const timestamp = typeof provider === 'number' ? provider : provider?.lastSync
    if (timestamp != null && Number.isFinite(timestamp) && timestamp > latest) latest = timestamp
  }
  return latest
}
