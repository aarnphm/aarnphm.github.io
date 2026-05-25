const javascriptCellMagicPattern = /^%%(?:javascript|js)\b/i
const captureCellMagicPattern = /^%%capture\b/i
const cellMagicPattern = /^%%([A-Za-z_][A-Za-z0-9_-]*)\b/

export type PreparedJavaScriptSource = { readonly source: string; readonly stripped: boolean }

export function stripJavaScriptCellMagics(source: string): PreparedJavaScriptSource {
  const lines = source.split(/\r?\n/)
  let index = 0
  let stripped = false

  while (index < lines.length) {
    const trimmed = lines[index].trim()
    if (trimmed.length === 0 && !stripped) {
      index += 1
      continue
    }
    if (captureCellMagicPattern.test(trimmed)) {
      index += 1
      stripped = true
      continue
    }
    if (javascriptCellMagicPattern.test(trimmed)) {
      index += 1
      stripped = true
      break
    }
    break
  }

  return stripped ? { source: lines.slice(index).join('\n'), stripped } : { source, stripped }
}

export function unsupportedJavaScriptRuntimeReason(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (trimmed.length === 0) continue
    if (captureCellMagicPattern.test(trimmed)) continue
    if (javascriptCellMagicPattern.test(trimmed)) return undefined
    const cellMagic = trimmed.match(cellMagicPattern)?.[1]
    if (cellMagic !== undefined) {
      return `%%${cellMagic} cells are unavailable in the JavaScript notebook runtime`
    }
    if (trimmed.startsWith('%'))
      return 'line magics are unavailable in the JavaScript notebook runtime'
    if (trimmed.startsWith('!'))
      return 'shell escapes are unavailable in the JavaScript notebook runtime'
    return undefined
  }
}
