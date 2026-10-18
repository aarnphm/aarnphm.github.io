// Watch mode imports a new build bundle for each source edit, and Node never unloads an imported
// ES module. A retired bundle keeps its module-level caches reachable, so the CLI calls
// releaseModuleCaches on it. Caches shared across bundles belong on globalThis instead.
const caches = new Set<{ clear(): void }>()

export function releasedOnReload<T extends { clear(): void }>(cache: T): T {
  caches.add(cache)
  return cache
}

export function onModuleRelease(callback: () => void): void {
  caches.add({ clear: callback })
}

export class ReloadableWeakMap<K extends WeakKey, V> {
  #map = new WeakMap<K, V>()

  constructor() {
    caches.add(this)
  }

  get(key: K): V | undefined {
    return this.#map.get(key)
  }

  set(key: K, value: V): this {
    this.#map.set(key, value)
    return this
  }

  has(key: K): boolean {
    return this.#map.has(key)
  }

  delete(key: K): boolean {
    return this.#map.delete(key)
  }

  clear(): void {
    this.#map = new WeakMap<K, V>()
  }
}

export function releaseModuleCaches(): void {
  for (const cache of caches) cache.clear()
}
