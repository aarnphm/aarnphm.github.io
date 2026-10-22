const LEAN_ASCII_TOKEN_LUT = new Uint8Array(128)
for (let c = 48; c <= 57; c += 1) LEAN_ASCII_TOKEN_LUT[c] = 1 // 0-9
for (let c = 65; c <= 90; c += 1) LEAN_ASCII_TOKEN_LUT[c] = 2 // A-Z
for (let c = 97; c <= 122; c += 1) LEAN_ASCII_TOKEN_LUT[c] = 2 // a-z
LEAN_ASCII_TOKEN_LUT[95] = 2 // _
LEAN_ASCII_TOKEN_LUT[35] = 3 // #
LEAN_ASCII_TOKEN_LUT[39] = 1 // '
LEAN_ASCII_TOKEN_LUT[33] = 1 // !
LEAN_ASCII_TOKEN_LUT[63] = 1 // ?
LEAN_ASCII_TOKEN_LUT[46] = 1 // .

function isLeanTokenCharCode(code: number): boolean {
  if (code < 128) return LEAN_ASCII_TOKEN_LUT[code] !== 0
  return (
    (code >= 0x00c0 && code <= 0x024f) ||
    (code >= 0x0370 && code <= 0x03ff) ||
    (code >= 0x2080 && code <= 0x2089) ||
    (code >= 0x2100 && code <= 0x214f)
  )
}

function isLeanIdentStartCharCode(code: number): boolean {
  if (code < 128) return LEAN_ASCII_TOKEN_LUT[code] === 2
  return (
    (code >= 0x00c0 && code <= 0x024f) ||
    (code >= 0x0370 && code <= 0x03ff) ||
    (code >= 0x2100 && code <= 0x214f)
  )
}

export function scanLeanPrefixBeforeColumn(
  lineText: string,
  col: number,
):
  | { readonly kind: 'backslash' }
  | { readonly kind: 'ident'; readonly fromCol: number; readonly prefix: string }
  | null {
  if (col <= 0) return null
  // Fast check for backslash token `\\[^\s\\]*` before `col`
  for (let i = col - 1; i >= 0; i -= 1) {
    const code = lineText.charCodeAt(i)
    if (code <= 32) break
    if (code === 92) {
      return { kind: 'backslash' }
    }
  }

  let start = col
  while (start > 0 && isLeanTokenCharCode(lineText.charCodeAt(start - 1))) {
    start -= 1
  }
  if (start === col) return null

  const firstCode = lineText.charCodeAt(start)
  if (firstCode === 35) {
    if (start + 1 >= col || !isLeanIdentStartCharCode(lineText.charCodeAt(start + 1))) {
      return null
    }
    return { kind: 'ident', fromCol: start, prefix: lineText.slice(start, col) }
  }

  while (start < col && !isLeanIdentStartCharCode(lineText.charCodeAt(start))) {
    start += 1
  }
  if (start >= col) return null
  return { kind: 'ident', fromCol: start, prefix: lineText.slice(start, col) }
}

export function extractLeanIdentAtColumn(
  lineText: string,
  col: number,
): { readonly token: string; readonly startCol: number; readonly endCol: number } | undefined {
  const len = lineText.length
  if (col < 0 || col > len || len === 0) return undefined
  let probe = col
  if (probe === len || !isLeanTokenCharCode(lineText.charCodeAt(probe))) {
    if (probe > 0 && isLeanTokenCharCode(lineText.charCodeAt(probe - 1))) {
      probe -= 1
    } else {
      return undefined
    }
  }
  let start = probe
  while (start > 0 && isLeanTokenCharCode(lineText.charCodeAt(start - 1))) {
    start -= 1
  }
  let end = probe + 1
  while (end < len && isLeanTokenCharCode(lineText.charCodeAt(end))) {
    end += 1
  }
  while (end > start && lineText.charCodeAt(end - 1) === 46) {
    end -= 1
  }
  if (end <= start) return undefined
  return { token: lineText.slice(start, end), startCol: start, endCol: end }
}
