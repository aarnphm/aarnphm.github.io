export type ProviderSync =
  | number
  | {
      readonly lastSync: number
      readonly sleepLastSync?: number
      readonly lactateThresholdLastSync?: number
    }
  | null
  | undefined

export function latestProviderSync(...providers: readonly ProviderSync[]): number {
  let latest = 0
  for (const provider of providers) {
    const timestamp = typeof provider === 'number' ? provider : provider?.lastSync
    if (timestamp != null && Number.isFinite(timestamp) && timestamp > latest) latest = timestamp
    const sleepTimestamp = typeof provider === 'object' ? provider?.sleepLastSync : null
    if (sleepTimestamp != null && Number.isFinite(sleepTimestamp) && sleepTimestamp > latest)
      latest = sleepTimestamp
    const lactateTimestamp =
      typeof provider === 'object' ? provider?.lactateThresholdLastSync : null
    if (lactateTimestamp != null && Number.isFinite(lactateTimestamp) && lactateTimestamp > latest)
      latest = lactateTimestamp
  }
  return latest
}
