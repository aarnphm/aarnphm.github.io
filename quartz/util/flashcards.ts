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

const separatorRe = /^-{3,}\s*$/
const clozeRe = /(?<!\[)\[(?!\[)([^[\]]+)\](?![\]()])/g

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

function renderCloze(
  sentence: string,
  matches: RegExpMatchArray[],
  target: number,
  face: 'front' | 'back',
): string {
  let out = ''
  let last = 0
  matches.forEach((match, i) => {
    const start = match.index ?? 0
    out += sentence.slice(last, start)
    const [answer, hint] = match[1].split('|').map(part => part.trim())
    if (i === target) {
      out +=
        face === 'front'
          ? `<span class="cloze-blank">${hint ?? '[…]'}</span>`
          : `<span class="cloze-answer">${answer}</span>`
    } else {
      out += answer
    }
    last = start + match[0].length
  })
  out += sentence.slice(last)
  return out
}

function makeClozeCards(sentence: string): Card[] {
  const matches = [...sentence.matchAll(clozeRe)]
  if (matches.length === 0) return []
  const groupId = hashCard(`C:${normalize(sentence)}`)
  return matches.map((match, index) => {
    const [answer, hint] = match[1].split('|').map(part => part.trim())
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
