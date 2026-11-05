export type Label = 'main' | 'quartz' | 'wrangler'
const colors: Record<Label, string> = {
  main: '\x1b[38;5;39m',
  quartz: '\x1b[38;5;110m',
  wrangler: '\x1b[38;5;214m',
}
const useColor = process.stdout.isTTY && process.stderr.isTTY
const ansiPattern = new RegExp(String.raw`\u001b\[[0-9;]*m`, 'g')
const prefix = (label: Label): string => {
  const text = `[${label.padEnd(8, ' ')}]`
  return useColor ? `${colors[label]}${text}\x1b[0m` : text
}

export const mainPrefix = prefix('main')

export function log(label: Label, message: string, stream: 'stdout' | 'stderr' = 'stdout'): void {
  const target = stream === 'stdout' ? process.stdout : process.stderr
  target.write(`${prefix(label)} ${message}\n`)
}

export function formatChunk(label: Label, chunk: Buffer): string {
  return chunk
    .toString()
    .split(/(?<=\n)/)
    .map(line => (line.length ? `${prefix(label)} ${line}` : line))
    .join('')
}

export function stripAnsi(value: string): string {
  return value.replace(ansiPattern, '')
}

export function describeError(value: unknown): string {
  if (value instanceof Error) return value.stack ?? value.message
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}
