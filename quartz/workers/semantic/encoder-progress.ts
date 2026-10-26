import type { ProgressInfo } from '@huggingface/transformers'
import type { ModelLoadProgress } from '../../util/semantic-protocol'

/** Reports bytes the library actually consumed, including reads from its model cache. */
export function encoderProgress(report: (progress: ModelLoadProgress) => void) {
  const files = new Map<string, { loaded: number; total: number; done: boolean }>()
  let lastReport = 0
  return (event: ProgressInfo) => {
    if (!('file' in event)) return
    const file = files.get(event.file) ?? { loaded: 0, total: 0, done: false }
    files.set(event.file, file)
    if (event.status === 'progress') {
      file.loaded = event.loaded
      file.total = event.total
    } else if (event.status === 'done') file.done = true
    const now = performance.now()
    if (event.status === 'progress' && now - lastReport < 100 && file.loaded < file.total) return
    lastReport = now
    const entries = [...files.values()]
    report({
      phase:
        [...files.keys()].some(name => name.includes('.onnx')) && entries.every(value => value.done)
          ? 'initializing'
          : 'loading',
      loadedBytes: entries.reduce((sum, value) => sum + value.loaded, 0),
      totalBytes: entries.reduce((sum, value) => sum + value.total, 0),
    })
  }
}
