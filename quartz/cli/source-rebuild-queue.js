import { setTimeout as delay } from 'node:timers/promises'

/** Own source paths until the build takes them under its mutex; retain failed batches for repair. */
export function createSourceRebuildQueue({
  build,
  onError,
  onReceipt = () => {},
  quietMs = 100,
  maxWaitMs = 500,
}) {
  const pending = new Set()
  const failed = new Set()
  let enabled = false
  let running
  let firstAt = 0
  let lastAt = 0
  let sequence = 0

  async function settle() {
    while (pending.size > 0) {
      const remaining = Math.min(lastAt + quietMs, firstAt + maxWaitMs) - performance.now()
      if (remaining <= 0) return
      await delay(remaining)
    }
  }

  async function drain(initial) {
    while (initial || pending.size > 0) {
      if (!initial) await settle()
      initial = false
      let consumed = []
      let taken = false
      const batch = ++sequence
      const startedAt = performance.now()
      try {
        const ready = await build(() => {
          if (taken) throw new Error('Source batch already taken')
          taken = true
          consumed = [...new Set([...failed, ...pending])]
          const queuedMs = pending.size > 0 ? performance.now() - firstAt : 0
          pending.clear()
          failed.clear()
          firstAt = 0
          lastAt = 0
          onReceipt({ type: 'batch', batch, at: Date.now(), paths: consumed, queuedMs })
          return consumed
        })
        if (!taken) throw new Error('Build did not take its source batch')
        if (ready === false) for (const fp of consumed) failed.add(fp)
        onReceipt({
          type: 'complete',
          batch,
          at: Date.now(),
          ready: ready !== false,
          elapsedMs: performance.now() - startedAt,
          pending: pending.size,
          retained: failed.size,
        })
      } catch (error) {
        for (const fp of consumed) failed.add(fp)
        onReceipt({ type: 'failed', batch, at: Date.now(), retained: failed.size })
        onError(error)
        // If acquisition failed before taking paths, keep them for the next source event.
        if (!taken) {
          for (const fp of pending) failed.add(fp)
          pending.clear()
        }
      }
    }
  }

  function kick(initial = false) {
    if (!enabled) return Promise.resolve()
    if (!running) {
      running = Promise.resolve()
        .then(() => drain(initial))
        .finally(() => {
          running = undefined
          if (pending.size > 0) void kick()
        })
    }
    return running
  }

  return {
    add(fp) {
      const now = performance.now()
      if (pending.size === 0) firstAt = now
      lastAt = now
      pending.add(fp)
      onReceipt({ type: 'queued', at: Date.now(), path: fp, pending: pending.size })
      return kick()
    },
    start() {
      if (enabled) return running ?? Promise.resolve()
      enabled = true
      return kick(true)
    },
  }
}
