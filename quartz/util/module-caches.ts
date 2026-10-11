// Watch mode imports a new build bundle for each source edit, and Node never unloads an imported
// ES module. A retired bundle keeps its module-level caches reachable, so the CLI calls
// releaseModuleCaches on it. Caches shared across bundles belong on globalThis instead.
const caches = new Set<{ clear(): void }>()

export function releasedOnReload<T extends { clear(): void }>(cache: T): T {
  caches.add(cache)
  return cache
}

export function releaseModuleCaches(): void {
  for (const cache of caches) cache.clear()
}
