import { createHash } from 'crypto'

export type CardKind = 'qa' | 'cloze'

export interface ClozeDeletion {
  index: number
  answer: string
  hint?: string
}

export interface Card {
  id: string
  kind: CardKind
  front: string
  back: string
  raw: string
  groupId?: string
  deletions?: ClozeDeletion[]
}

export interface DeckError {
  line: number
  message: string
}

export interface Deck {
  cards: Card[]
  errors: DeckError[]
}

type MathDelimiter = '$' | '$$'

interface ClozeMatch {
  start: number
  end: number
  value: string
  mathDelimiter?: MathDelimiter
  leftMath?: boolean
  rightMath?: boolean
}

interface ClozeParts {
  answer: string
  hint?: string
}

const separatorRe = /^-{3,}\s*$/

export function hashCard(canonical: string): string {
  return createHash('sha256').update(canonical).digest('hex').slice(0, 8)
}

function normalize(text: string): string {
  return text.trim().replace(/\s+/g, ' ')
}

function stripFrontmatter(source: string): { body: string; offset: number } {
  const lines = source.split(/\r?\n/)
  if (lines[0]?.trim() === '---') {
    for (let i = 1; i < lines.length; i++) {
      if (lines[i]?.trim() === '---') {
        return { body: lines.slice(i + 1).join('\n'), offset: i + 1 }
      }
    }
  }
  return { body: source, offset: 0 }
}

function makeQaCard(front: string, back: string): Card {
  return {
    id: hashCard(`Q:${normalize(front)} A:${normalize(back)}`),
    kind: 'qa',
    front,
    back,
    raw: `Q: ${front}\nA: ${back}`,
  }
}

function isEscaped(source: string, index: number): boolean {
  let backslashes = 0
  for (let i = index - 1; i >= 0 && source[i] === '\\'; i--) backslashes++
  return backslashes % 2 === 1
}

function mathDelimiterAt(source: string, index: number): MathDelimiter | undefined {
  if (source[index] !== '$' || isEscaped(source, index)) return undefined
  return source[index + 1] === '$' ? '$$' : '$'
}

function splitClozeValue(value: string): ClozeParts {
  const separator = value.indexOf('|')
  if (separator === -1) return { answer: value.trim() }
  return { answer: value.slice(0, separator).trim(), hint: value.slice(separator + 1).trim() }
}

function findMathClose(source: string, from: number, delimiter: MathDelimiter): number {
  for (let i = from; i < source.length; i++) {
    if (mathDelimiterAt(source, i) === delimiter) return i
  }
  return -1
}

function findClozeMatches(sentence: string): ClozeMatch[] {
  const matches: ClozeMatch[] = []
  let mathDelimiter: MathDelimiter | undefined
  let mathOpen = -1

  for (let i = 0; i < sentence.length; i++) {
    const delimiter = mathDelimiterAt(sentence, i)
    if (delimiter) {
      mathDelimiter = mathDelimiter === delimiter ? undefined : delimiter
      if (mathDelimiter === delimiter) mathOpen = i
      i += delimiter.length - 1
      continue
    }

    if (sentence[i] !== '[') continue
    if (sentence[i - 1] === '[' || sentence[i + 1] === '[') continue

    const close = sentence.indexOf(']', i + 1)
    if (close === -1) break

    const value = sentence.slice(i + 1, close)
    const after = sentence[close + 1]
    if (value.length === 0 || value.includes('[')) continue
    if (after === ']' || after === '(' || after === ')') continue

    if (!mathDelimiter) {
      matches.push({ start: i, end: close + 1, value })
      i = close
      continue
    }

    const region = mathDelimiter
    const len = region.length
    const leftMath = sentence.slice(mathOpen + len, i).trim().length > 0
    const closeIdx = findMathClose(sentence, close + 1, region)
    const rightMath = closeIdx === -1 || sentence.slice(close + 1, closeIdx).trim().length > 0
    const start = leftMath ? i : mathOpen
    const end = rightMath ? close + 1 : closeIdx + len
    if (!rightMath) mathDelimiter = undefined
    matches.push({ start, end, value, mathDelimiter: region, leftMath, rightMath })
    i = end - 1
  }

  return matches
}

function renderClozeReplacement(
  match: ClozeMatch,
  target: boolean,
  face: 'front' | 'back',
): string {
  const { answer, hint } = splitClozeValue(match.value)
  const delimiter = match.mathDelimiter
  if (!target) {
    if (!delimiter) return answer
    const open = match.leftMath ? '' : delimiter
    const shut = match.rightMath ? '' : delimiter
    return `${open}${answer}${shut}`
  }
  if (!delimiter) {
    return face === 'front'
      ? `<span class="cloze-blank">${hint ?? '[…]'}</span>`
      : `<span class="cloze-answer">${answer}</span>`
  }

  const left = match.leftMath ? delimiter : ''
  const right = match.rightMath ? delimiter : ''
  return face === 'front'
    ? `${left}<span class="cloze-blank">${hint ?? '[…]'}</span>${right}`
    : `${left}<span class="cloze-answer">${delimiter}${answer}${delimiter}</span>${right}`
}

function renderCloze(
  sentence: string,
  matches: ClozeMatch[],
  target: number,
  face: 'front' | 'back',
): string {
  let out = ''
  let last = 0
  matches.forEach((match, i) => {
    out += sentence.slice(last, match.start)
    out += renderClozeReplacement(match, i === target, face)
    last = match.end
  })
  out += sentence.slice(last)
  return out
}

function makeClozeCards(sentence: string): Card[] {
  const matches = findClozeMatches(sentence)
  if (matches.length === 0) return []
  const groupId = hashCard(`C:${normalize(sentence)}`)
  return matches.map((match, index) => {
    const { answer, hint } = splitClozeValue(match.value)
    return {
      id: hashCard(`C:${normalize(sentence)} ${index}`),
      kind: 'cloze' as const,
      front: renderCloze(sentence, matches, index, 'front'),
      back: renderCloze(sentence, matches, index, 'back'),
      raw: `C: ${sentence}`,
      groupId,
      deletions: [{ index, answer, hint }],
    }
  })
}

interface Pending {
  kind: CardKind
  startLine: number
  q: string[]
  a: string[]
  c: string[]
  sawAnswer: boolean
}

export function parseFlashcards(source: string): Deck {
  const { body, offset } = stripFrontmatter(source)
  const lines = body.split(/\r?\n/)
  const cards: Card[] = []
  const errors: DeckError[] = []
  let cur: Pending | null = null

  const lineNo = (index: number) => index + offset + 1

  const flush = () => {
    if (!cur) return
    if (cur.kind === 'qa') {
      const front = cur.q.join('\n').trim()
      const back = cur.a.join('\n').trim()
      if (!cur.sawAnswer || back.length === 0) {
        errors.push({ line: lineNo(cur.startLine), message: 'Q: card missing A:' })
      } else {
        cards.push(makeQaCard(front, back))
      }
    } else {
      const sentence = cur.c.join('\n').trim()
      const siblings = makeClozeCards(sentence)
      if (siblings.length === 0) {
        errors.push({ line: lineNo(cur.startLine), message: 'C: card missing [deletions]' })
      } else {
        cards.push(...siblings)
      }
    }
    cur = null
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const qMatch = /^\s*Q:(.*)$/.exec(line)
    const aMatch = /^\s*A:(.*)$/.exec(line)
    const cMatch = /^\s*C:(.*)$/.exec(line)

    if (separatorRe.test(line.trim())) {
      flush()
      continue
    }
    if (qMatch) {
      flush()
      cur = {
        kind: 'qa',
        startLine: i,
        q: [qMatch[1].replace(/^ /, '')],
        a: [],
        c: [],
        sawAnswer: false,
      }
      continue
    }
    if (cMatch) {
      flush()
      cur = {
        kind: 'cloze',
        startLine: i,
        q: [],
        a: [],
        c: [cMatch[1].replace(/^ /, '')],
        sawAnswer: false,
      }
      continue
    }
    if (aMatch && cur?.kind === 'qa' && !cur.sawAnswer) {
      cur.sawAnswer = true
      cur.a.push(aMatch[1].replace(/^ /, ''))
      continue
    }
    if (!cur) continue
    if (cur.kind === 'cloze') cur.c.push(line)
    else if (cur.sawAnswer) cur.a.push(line)
    else cur.q.push(line)
  }
  flush()

  return { cards, errors }
}
