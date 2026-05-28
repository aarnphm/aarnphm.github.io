import { styleText } from 'node:util'
import { isMainThread } from 'node:worker_threads'
import pretty from 'pretty-time'

type BuildSpanArgv = { verbose: boolean; slowBuildThreshold?: number; allBuildSpans?: boolean }
type SlowBuildSpan = { elapsedMs: number; label: string; subject: string }
type SlowBuildSpanGroup = { count: number; max: SlowBuildSpan; totalMs: number }

declare global {
  var __quartzSlowBuildSpans: SlowBuildSpan[] | undefined
}

const slowBuildSpans = (globalThis.__quartzSlowBuildSpans ??= [])

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
  if (argv.verbose || argv.allBuildSpans) return true
  const threshold = slowBuildThresholdMs(argv)
  return threshold !== undefined && elapsedMs >= threshold
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
  if (
    isMainThread &&
    !argv.verbose &&
    !argv.allBuildSpans &&
    slowBuildThresholdMs(argv) !== undefined
  ) {
    slowBuildSpans.push({ elapsedMs, label, subject })
    return
  }
  console.log(`[${label}] ${subject} (${formatDurationMs(elapsedMs)})`)
}

function formatSlowSpan(span: SlowBuildSpan): string {
  return `[${span.label}] ${span.subject} (${formatDurationMs(span.elapsedMs)})`
}

export function flushBuildSpans(argv: BuildSpanArgv): void {
  if (
    argv.verbose ||
    argv.allBuildSpans ||
    slowBuildThresholdMs(argv) === undefined ||
    slowBuildSpans.length === 0
  ) {
    slowBuildSpans.length = 0
    return
  }

  const groups = new Map<string, SlowBuildSpanGroup>()
  for (const span of slowBuildSpans) {
    const group = groups.get(span.label)
    if (group) {
      group.count += 1
      group.totalMs += span.elapsedMs
      if (span.elapsedMs > group.max.elapsedMs) {
        group.max = span
      }
    } else {
      groups.set(span.label, { count: 1, max: span, totalMs: span.elapsedMs })
    }
  }

  console.log(`[slow] ${slowBuildSpans.length} spans >= ${slowBuildThresholdMs(argv)}ms`)
  const orderedGroups = Array.from(groups.entries()).sort(
    ([, left], [, right]) => right.totalMs - left.totalMs,
  )
  for (const [label, group] of orderedGroups.slice(0, 12)) {
    console.log(
      `[slow:${label}] ${group.count} spans, total ${formatDurationMs(group.totalMs)}, max ${formatSlowSpan(group.max)}`,
    )
  }

  const topSpans = [...slowBuildSpans]
    .sort((left, right) => right.elapsedMs - left.elapsedMs)
    .slice(0, 20)
  for (const span of topSpans) {
    console.log(`[slow:top] ${formatSlowSpan(span)}`)
  }

  slowBuildSpans.length = 0
}
