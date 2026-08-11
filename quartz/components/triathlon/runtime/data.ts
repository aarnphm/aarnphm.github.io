export type ResourceResult<T> =
  | { status: 'ready'; value: T }
  | { status: 'aborted' }
  | { status: 'error'; error: Error }

export interface NavigationResource<T> {
  load(path: string): Promise<ResourceResult<T>>
  peek(path: string): ResourceResult<T> | undefined
}

const resourceError = (error: unknown): Error =>
  error instanceof Error ? error : new Error(String(error))

export const createNavigationResource = <T>(
  signal: AbortSignal,
  read: (response: Response) => Promise<T>,
): NavigationResource<T> => {
  const pending = new Map<string, Promise<ResourceResult<T>>>()
  const settled = new Map<string, ResourceResult<T>>()
  const load = (path: string): Promise<ResourceResult<T>> => {
    const cached = settled.get(path)
    if (cached?.status === 'ready') return Promise.resolve(cached)
    const current = pending.get(path)
    if (current) return current
    if (signal.aborted) return Promise.resolve({ status: 'aborted' })
    const request = fetch(path, { signal })
      .then(async response => {
        if (!response.ok) throw new Error(`${path} returned ${response.status}`)
        const result: ResourceResult<T> = { status: 'ready', value: await read(response) }
        settled.set(path, result)
        return result
      })
      .catch((error: unknown) => {
        const result: ResourceResult<T> =
          signal.aborted || (error instanceof DOMException && error.name === 'AbortError')
            ? { status: 'aborted' }
            : { status: 'error', error: resourceError(error) }
        settled.set(path, result)
        return result
      })
      .finally(() => pending.delete(path))
    pending.set(path, request)
    return request
  }
  return { load, peek: path => settled.get(path) }
}
