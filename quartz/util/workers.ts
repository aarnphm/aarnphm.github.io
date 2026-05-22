export const workerEntryPattern = 'quartz/**/*.worker.ts'

export function isWorkerEntryPath(path: string): boolean {
  return path.endsWith('.worker.ts')
}
