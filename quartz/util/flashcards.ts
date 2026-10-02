import { createHash } from 'crypto'
import { stripSpeechMarkup } from '../extensions/micromark-extension-speech/source'

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
  /** Back-only `N:` text; excluded from the id so notes never reset scheduling. */
  note?: string
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
  /** Source text kept around the answer when this deletion is not the tested one. */
  before: string
  after: string
  /** Math or code delimiter that wraps a revealed answer, empty for plain text. */
  wrap: string
  /** Text that closes and reopens a math or code region around the tested deletion. */
  open: string
  shut: string
}

interface CodeSpan {
  fence: string
  open: number
  close: number
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
  return stripSpeechMarkup(text).trim().replace(/\s+/g, ' ')
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

function backtickRun(source: string, index: number): number {
  let end = index
  while (source[end] === '`') end++
  return end - index
}

/** CommonMark code span opening at `index`, or undefined when no run of the same length closes it. */
function codeSpanAt(source: string, index: number): CodeSpan | undefined {
  const length = backtickRun(source, index)
  for (let i = index + length; i < source.length; ) {
    const run = backtickRun(source, i)
    if (run === length) return { fence: source.slice(index, index + length), open: index, close: i }
    i += Math.max(run, 1)
  }
  return undefined
}

/**
 * Splitting a code span moves the whitespace between the deletion and the remaining code outside the
 * span. Remaining code that would touch the new fence with its own backtick keeps one padding space.
 */
function codeEdge(text: string, side: 'start' | 'end'): { gap: string; pad: string } {
  const gap = (side === 'end' ? /\s*$/ : /^\s*/).exec(text)![0]
  const rest = side === 'end' ? text.slice(0, text.length - gap.length) : text.slice(gap.length)
  const edge = side === 'end' ? rest.at(-1) : rest[0]
  return { gap, pad: edge === '`' ? ' ' : '' }
}

function findClozeMatches(sentence: string): ClozeMatch[] {
  const matches: ClozeMatch[] = []
  let mathDelimiter: MathDelimiter | undefined
  let mathOpen = -1
  let code: CodeSpan | undefined

  const push = (
    start: number,
    bracket: number,
    close: number,
    end: number,
    region: Pick<ClozeMatch, 'wrap' | 'open' | 'shut'>,
  ) =>
    matches.push({
      start,
      end,
      value: sentence.slice(bracket + 1, close),
      before: sentence.slice(start, bracket),
      after: sentence.slice(close + 1, end),
      ...region,
    })

  for (let i = 0; i < sentence.length; i++) {
    if (code && i === code.close) {
      i += code.fence.length - 1
      code = undefined
      continue
    }
    if (!code && !mathDelimiter && sentence[i] === '`' && !isEscaped(sentence, i)) {
      code = codeSpanAt(sentence, i)
      i += backtickRun(sentence, i) - 1
      continue
    }

    const delimiter = code ? undefined : mathDelimiterAt(sentence, i)
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

    if (code) {
      if (close > code.close) continue
      const { fence } = code
      const lastEnd = matches.at(-1)?.end ?? 0
      const leftText = sentence.slice(Math.max(code.open + fence.length, lastEnd), i)
      const rightText = sentence.slice(close + 1, code.close)
      const leftCode = sentence.slice(code.open + fence.length, i).trim().length > 0
      const rightCode = rightText.trim().length > 0
      const left = codeEdge(leftText, 'end')
      const right = codeEdge(rightText, 'start')
      const start = leftCode ? i - left.gap.length : code.open
      const end = rightCode ? close + 1 + right.gap.length : code.close + fence.length
      push(start, i, close, end, {
        wrap: fence,
        open: leftCode ? `${left.pad}${fence}${left.gap}` : '',
        shut: rightCode ? `${right.gap}${fence}${right.pad}` : '',
      })
      if (!rightCode) code = undefined
      i = end - 1
      continue
    }

    if (!mathDelimiter) {
      push(i, i, close, close + 1, { wrap: '', open: '', shut: '' })
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
    push(start, i, close, end, {
      wrap: region,
      open: leftMath ? region : '',
      shut: rightMath ? region : '',
    })
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
  if (!target) return `${match.before}${answer}${match.after}`
  const { wrap, open, shut } = match
  return face === 'front'
    ? `${open}<span class="cloze-blank">${hint ?? '[…]'}</span>${shut}`
    : `${open}<span class="cloze-answer">${wrap}${answer}${wrap}</span>${shut}`
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
  n: string[]
  sawAnswer: boolean
  sawNote: boolean
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
    const note = cur.n.join('\n').trim()
    const withNote = (card: Card): Card => (note ? { ...card, note } : card)
    if (cur.kind === 'qa') {
      const front = cur.q.join('\n').trim()
      const back = cur.a.join('\n').trim()
      if (!cur.sawAnswer || back.length === 0) {
        errors.push({ line: lineNo(cur.startLine), message: 'Q: card missing A:' })
      } else {
        cards.push(withNote(makeQaCard(front, back)))
      }
    } else {
      const sentence = cur.c.join('\n').trim()
      const siblings = makeClozeCards(sentence)
      if (siblings.length === 0) {
        errors.push({ line: lineNo(cur.startLine), message: 'C: card missing [deletions]' })
      } else {
        cards.push(...siblings.map(withNote))
      }
    }
    cur = null
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const qMatch = /^\s*Q:(.*)$/.exec(line)
    const aMatch = /^\s*A:(.*)$/.exec(line)
    const cMatch = /^\s*C:(.*)$/.exec(line)
    const nMatch = /^\s*N:(.*)$/.exec(line)

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
        n: [],
        sawAnswer: false,
        sawNote: false,
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
        n: [],
        sawAnswer: false,
        sawNote: false,
      }
      continue
    }
    if (aMatch && cur?.kind === 'qa' && !cur.sawAnswer) {
      cur.sawAnswer = true
      cur.a.push(aMatch[1].replace(/^ /, ''))
      continue
    }
    if (nMatch && cur && !cur.sawNote) {
      if (cur.kind === 'qa' && !cur.sawAnswer) {
        errors.push({ line: lineNo(i), message: 'N: before A:' })
        cur = null
        continue
      }
      cur.sawNote = true
      cur.n.push(nMatch[1].replace(/^ /, ''))
      continue
    }
    if (!cur) continue
    if (cur.sawNote) cur.n.push(line)
    else if (cur.kind === 'cloze') cur.c.push(line)
    else if (cur.sawAnswer) cur.a.push(line)
    else cur.q.push(line)
  }
  flush()

  return { cards, errors }
}
