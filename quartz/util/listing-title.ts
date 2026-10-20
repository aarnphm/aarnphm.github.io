const TITLE_QUOTE_PAIRS = [
  ['"', '"'],
  ['\u201C', '\u201D'],
  ['\u201F', '\u201D'],
  ['\u00AB', '\u00BB'],
] as const

export function unwrapTitleQuotes(title: string): string {
  let current = title.trim()
  for (;;) {
    const pair = TITLE_QUOTE_PAIRS.find(
      ([open, close]) =>
        current.length > open.length + close.length &&
        current.startsWith(open) &&
        current.endsWith(close),
    )
    if (!pair) return current
    current = current.slice(pair[0].length, current.length - pair[1].length).trim()
  }
}
