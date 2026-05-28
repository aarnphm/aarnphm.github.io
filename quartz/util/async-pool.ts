import { availableParallelism } from 'node:os'

const parallelism = availableParallelism()

export const defaultEmitterConcurrency = Math.min(Math.max(parallelism, 4), 8)
export const defaultIoConcurrency = Math.min(Math.max(parallelism * 2, 8), 24)

export async function mapConcurrent<T, R>(
  items: readonly T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const workerCount = Math.min(Math.max(Math.floor(concurrency), 1), items.length)
  const results: R[] = []
  let nextIndex = 0

  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex
      nextIndex += 1
      results[index] = await fn(items[index], index)
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()))
  return results
}
