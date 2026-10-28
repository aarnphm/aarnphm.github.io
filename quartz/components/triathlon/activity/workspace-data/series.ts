import type { WorkspaceAxis, WorkspaceSample, WorkspaceTrace } from '../workspace-data'

export const workspacePosition = (sample: WorkspaceSample, axis: WorkspaceAxis): number =>
  axis === 'time' ? sample.elapsedS : sample.distanceKm

export const workspaceTraceDomain = (trace: WorkspaceTrace): [number, number] => {
  if (trace.domain) return trace.domain
  let low = Infinity
  let high = -Infinity
  for (const sample of trace.samples) {
    if (sample.value == null || !Number.isFinite(sample.value)) continue
    low = Math.min(low, sample.value)
    high = Math.max(high, sample.value)
  }
  return [low, high]
}

export const workspaceTraceY = (value: number, [low, high]: readonly [number, number]): number =>
  high === low ? 50 : ((high - value) / (high - low)) * 100

export const workspaceTracePaths = (
  trace: WorkspaceTrace,
  axis: WorkspaceAxis,
  start: number,
  end: number,
): { line: string; area: string } => {
  const domain = workspaceTraceDomain(trace)
  let line = ''
  let area = ''
  let segment = ''
  let firstX = 0
  let lastX = 0
  let previous: (WorkspaceSample & { value: number }) | null = null
  const close = (): void => {
    if (!segment) return
    line += `${segment} `
    area += `${segment} L ${lastX} 100 L ${firstX} 100 Z `
    segment = ''
  }
  const append = (position: number, value: number): void => {
    const x = Number((((position - start) / Math.max(end - start, 0.001)) * 100).toFixed(3))
    const y = workspaceTraceY(value, domain).toFixed(3)
    if (!segment) firstX = x
    segment += `${segment ? ' L' : 'M'} ${x} ${y}`
    lastX = x
  }
  for (const sample of trace.samples) {
    const position = workspacePosition(sample, axis)
    if (sample.value == null || !Number.isFinite(sample.value)) {
      close()
      previous = null
      continue
    }
    const value = sample.value
    if (previous) {
      const leftSample = previous
      const previousPosition = workspacePosition(leftSample, axis)
      if (position >= start && previousPosition <= end) {
        const valueAt = (boundary: number): number => {
          if (boundary === position) return value
          const fraction = (boundary - previousPosition) / (position - previousPosition)
          return leftSample.value + fraction * (value - leftSample.value)
        }
        if (!segment) {
          const left = Math.max(start, previousPosition)
          append(left, left === previousPosition ? previous.value : valueAt(left))
        }
        const right = Math.min(end, position)
        append(right, valueAt(right))
      }
    } else if (position >= start && position <= end) {
      append(position, sample.value)
    }
    if (position > end) close()
    previous = { ...sample, value: sample.value }
  }
  close()
  return { line, area }
}

export const workspaceValueAt = (trace: WorkspaceTrace, elapsedS: number): number | null => {
  const samples = trace.samples
  if (elapsedS < samples[0].elapsedS || elapsedS > samples[samples.length - 1].elapsedS) return null
  let low = 0
  let high = samples.length - 1
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (samples[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const right = samples[low]
  const left = samples[Math.max(0, low - 1)]
  if (right.elapsedS === elapsedS) return right.value
  if (left.value == null || right.value == null) return null
  return elapsedS - left.elapsedS <= right.elapsedS - elapsedS ? left.value : right.value
}
