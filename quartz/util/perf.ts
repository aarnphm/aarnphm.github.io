import { styleText } from 'node:util'
import pretty from 'pretty-time'

type BuildSpanArgv = { verbose: boolean; slowBuildThreshold?: number }

export class PerfTimer {
  evts: { [key: string]: [number, number] }

  constructor() {
    this.evts = {}
    this.addEvent('start')
  }

  addEvent(evtName: string) {
    this.evts[evtName] = process.hrtime()
  }

  timeSince(evtName?: string): string {
    return styleText('yellow', pretty(process.hrtime(this.evts[evtName ?? 'start'])))
  }

  elapsedMs(evtName?: string): number {
    const [seconds, nanoseconds] = process.hrtime(this.evts[evtName ?? 'start'])
    return seconds * 1000 + nanoseconds / 1_000_000
  }
}

export function slowBuildThresholdMs(argv: Pick<BuildSpanArgv, 'slowBuildThreshold'>) {
  const threshold = argv.slowBuildThreshold
  if (typeof threshold !== 'number') return undefined
  if (!Number.isFinite(threshold) || threshold <= 0) return undefined
  return threshold
}

export function shouldLogBuildSpan(argv: BuildSpanArgv, elapsedMs: number): boolean {
  const threshold = slowBuildThresholdMs(argv)
  return threshold === undefined ? argv.verbose : elapsedMs >= threshold
}

export function formatDurationMs(elapsedMs: number): string {
  const seconds = Math.trunc(elapsedMs / 1000)
  const nanoseconds = Math.round((elapsedMs - seconds * 1000) * 1_000_000)
  const duration: [number, number] = [seconds, nanoseconds]
  return styleText('yellow', pretty(duration))
}

export function logBuildSpan(
  argv: BuildSpanArgv,
  label: string,
  subject: string,
  elapsedMs: number,
): void {
  if (!shouldLogBuildSpan(argv, elapsedMs)) return
  console.log(`[${label}] ${subject} (${formatDurationMs(elapsedMs)})`)
}
