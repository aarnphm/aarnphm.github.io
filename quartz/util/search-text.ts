const CONTEXT_WINDOW_WORDS = 30

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function tokenizeTerm(term: string): string[] {
  const tokens = term.split(/\s+/).filter(token => token.length > 0)
  const tokenCount = tokens.length
  for (let index = 1; index < tokenCount; index++) {
    tokens.push(tokens.slice(0, index + 1).join(' '))
  }
  return tokens.sort((left, right) => right.length - left.length)
}

export function highlight(searchTerm: string, text: string, trim = false): string {
  const terms = tokenizeTerm(searchTerm)
  const words = text.split(/\s+/).filter(Boolean)
  const wordCount = words.length
  let startIndex = 0
  let endIndex = wordCount - 1

  if (trim && wordCount > 0) {
    const prefixes = terms.map(term => term.toLowerCase())
    const matches = words.map(word => {
      const lower = word.toLowerCase()
      return prefixes.some(term => lower.startsWith(term))
    })
    let bestSum = 0
    let bestIndex = 0
    let sum = matches
      .slice(0, CONTEXT_WINDOW_WORDS)
      .reduce((total, match) => total + Number(match), 0)
    const finalWindowStart = Math.max(wordCount - CONTEXT_WINDOW_WORDS, 0)
    for (let index = 0; index <= finalWindowStart; index++) {
      if (sum >= bestSum) {
        bestSum = sum
        bestIndex = index
      }
      sum += Number(matches[index + CONTEXT_WINDOW_WORDS] ?? false) - Number(matches[index])
    }
    startIndex = Math.max(bestIndex - CONTEXT_WINDOW_WORDS, 0)
    endIndex = Math.min(startIndex + 2 * CONTEXT_WINDOW_WORDS, wordCount - 1)
  }

  const highlighted = words
    .slice(startIndex, endIndex + 1)
    .map(word => {
      let output = word
      for (const term of terms) {
        if (!word.toLowerCase().includes(term.toLowerCase())) continue
        output = output.replace(
          new RegExp(escapeRegex(term), 'gi'),
          '<span class="highlight">$&</span>',
        )
      }
      return output
    })
    .join(' ')

  return `${startIndex === 0 ? '' : '...'}${highlighted}${endIndex >= wordCount - 1 ? '' : '...'}`
}

export function encode(value: string): string[] {
  const tokens: string[] = []
  let bufferStart = -1
  let bufferEnd = -1
  const lower = value.toLowerCase()

  let index = 0
  for (const character of lower) {
    const code = character.codePointAt(0)
    if (code === undefined) continue
    const isCjk =
      (code >= 0x3040 && code <= 0x309f) ||
      (code >= 0x30a0 && code <= 0x30ff) ||
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0xac00 && code <= 0xd7af) ||
      (code >= 0x20000 && code <= 0x2a6df)
    const isWhitespace = code === 32 || code === 9 || code === 10 || code === 13

    if (isCjk) {
      if (bufferStart !== -1) {
        tokens.push(lower.slice(bufferStart, bufferEnd))
        bufferStart = -1
      }
      tokens.push(character)
    } else if (isWhitespace) {
      if (bufferStart !== -1) {
        tokens.push(lower.slice(bufferStart, bufferEnd))
        bufferStart = -1
      }
    } else {
      if (bufferStart === -1) bufferStart = index
      bufferEnd = index + character.length
    }

    index += character.length
  }

  if (bufferStart !== -1) tokens.push(lower.slice(bufferStart))
  return tokens
}

// encode splits on whitespace only, so `lectures/2/afp` is one token and only its prefix matches.
// Index the full path, each folder segment (`03-varint`), and each word (`varint`).
export function pathSearchText(path: string): string {
  const tokens = [path, ...path.split('/'), ...path.split(/[/_.-]+/)]
  return [...new Set(tokens.filter(token => token.length > 0))].join(' ')
}
