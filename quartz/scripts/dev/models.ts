import { watch, type FSWatcher } from 'node:fs'
import path from 'node:path'
import { syncLatestLocalPaceModels } from '../local-pace-models'
import { describeError, log } from './log'

export function createModelArchiveWatcher(root: string, isReady: () => boolean) {
  const archive = path.join(root, '.models-archive')
  const output = path.join(root, 'public')
  let watcher: FSWatcher | undefined
  let retry: ReturnType<typeof setTimeout> | undefined
  let timer: ReturnType<typeof setTimeout> | undefined
  let disposed = false
  const sync = () => {
    if (!isReady()) return
    try {
      const selections = syncLatestLocalPaceModels(archive, output)
      if (selections.length)
        log(
          'main',
          `local models ${selections.map(({ family, archive }) => `${family}=${archive}`).join(' ')}`,
        )
    } catch (error) {
      log('main', `local model sync failed: ${describeError(error)}`, 'stderr')
    }
  }
  const schedule = () => {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      timer = undefined
      sync()
    }, 100)
  }
  const scheduleRetry = () => {
    retry = setTimeout(() => {
      retry = undefined
      start()
    }, 1000)
  }
  const start = () => {
    if (disposed || watcher) return
    try {
      const current = watch(archive, { recursive: true }, schedule)
      watcher = current
      current.once('error', error => {
        current.close()
        if (watcher === current) watcher = undefined
        if (!disposed) {
          log('main', `local model watcher failed: ${describeError(error)}`, 'stderr')
          scheduleRetry()
        }
      })
    } catch (error) {
      if (
        !(typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT')
      ) {
        log('main', `local model watcher failed: ${describeError(error)}`, 'stderr')
        return
      }
      scheduleRetry()
    }
  }
  return {
    start,
    sync,
    dispose() {
      disposed = true
      watcher?.close()
      watcher = undefined
      if (retry) clearTimeout(retry)
      if (timer) clearTimeout(timer)
    },
  }
}
