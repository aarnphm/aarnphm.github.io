import { effortCues, effortCueSegments } from './effort-cues'

export type SwimEffort = 'easy' | 'technique' | 'steady' | 'build' | 'hard'

export interface SwimPart {
  meters: number | null
  seconds: number | null
  effort: SwimEffort | null
}

export interface SwimRest {
  min: number
  max: number | null
  /** The coach wrote a send-off ("on 30"): each rep leaves on this clock. */
  interval: boolean
}

export interface SwimStep {
  label: string
  reps: number
  /** One rep; a split rep ("25m drill/25m swim") has several parts. */
  parts: SwimPart[]
  rest: SwimRest | null
  notes: string[]
}

export type SwimBlock =
  | { kind: 'set'; repeat: number; steps: SwimStep[]; rest: SwimRest | null }
  | { kind: 'rest'; rest: SwimRest }

/** A swim session read from the coach's free-text description. */
export interface SwimStructure {
  blocks: SwimBlock[]
  meters: number
  seconds: number
  /** The chart axis; null when the description mixes distance and time. */
  unit: 'meters' | 'seconds' | null
}

// The first match wins, so the order resolves lines such as "very easy swim down including kick".
const EFFORTS: [SwimEffort, RegExp][] = [
  ['hard', /\b(?:hard|fast|sprint|max(?:imal)?|race)\b/i],
  ['build', /\b(?:build\w*|tempo|accelerat\w*|desc\w*|threshold|negative)\b/i],
  ['easy', /\b(?:easy|warm[\s-]?up|loosen|swim down|cool[\s-]?down|float|recovery|choice)\b/i],
  [
    'technique',
    /\b(?:drill|kick|doggy|lead arm|fingertip|single arm|catch[\s-]?up|scull\w*|tech\w*)\b/i,
  ],
  ['steady', /\b(?:steady|non-?stop|moderate|aerobic|pull|paddles|swim)\b/i],
]

const UNIT = String.raw`(m|min|mins|minutes?|sec|secs|seconds?|s)?(?![\w])`
const TIME_UNIT = String.raw`(sec|secs|seconds?|s|min|mins|minutes?)`
const STEP = new RegExp(
  String.raw`^(?:(\d+(?:\s*-\s*\d+)*)\s*[x×]\s*)?(\d+(?:\.\d+)?)\s*${UNIT}\s*(.*)$`,
  'i',
)
const PIECE = new RegExp(String.raw`^(\d+(?:\.\d+)?)\s*${UNIT}\s*(.*)$`, 'i')
const REST_LINE = new RegExp(
  String.raw`^(?:rest\s*)?(\d+)(?:\s*-\s*(\d+))?\s*${TIME_UNIT}\.?(?:\s+rest)?\.?$`,
  'i',
)
const ROUND_REST = new RegExp(
  String.raw`^extra\s+(\d+)\s*${TIME_UNIT}\s+between\s+(?:sets|rounds|blocks)\.?$`,
  'i',
)
const NESTED = /^(\d+)\s*[x×]\s*\((.+)\)$/i
// "2(2 x 50) superman" repeats the bracketed set; the text after it describes the reps.
const NESTED_BARE = /^(\d+)\s*\((\d+\s*[x×]\s*[^()]+)\)\s*(.*)$/i
const ALTERNATION = new RegExp(
  String.raw`^(\d+)\s+([a-z]+)(?:\s*\((\d+)\s*${TIME_UNIT}\))?\s*,\s*(\d+)\s+([a-z]+)(?:\s*\((\d+)\s*${TIME_UNIT}\))?\s*,?\s*repeat\.?$`,
  'i',
)
const REST_TEXT: [RegExp, boolean][] = [
  [new RegExp(String.raw`\((\d+)\s*${TIME_UNIT}\b[^)]*\)`, 'i'), false],
  [new RegExp(String.raw`\b(?:leaving\s+)?on\s+(\d+)\s*${TIME_UNIT}?\s*rest\b`, 'i'), false],
  [new RegExp(String.raw`\bon\s+(\d+)\s*${TIME_UNIT}?(?![\w])`, 'i'), true],
  [
    new RegExp(
      String.raw`\brest\s+(\d+)\s*${TIME_UNIT}?(?:\s+after\s+each(?:\s+rep)?)?(?![\w])`,
      'i',
    ),
    false,
  ],
]

const seconds = (value: number, unit: string | undefined): number =>
  unit && /^min/i.test(unit) ? value * 60 : value

// An RPE out of 10 ("7/10", "RPE 8") names the effort before any keyword; the hardest one wins.
const rpeEffort = (text: string): SwimEffort | null => {
  const values = effortCues(text).flatMap(cue =>
    /\/\s*10\b|^RPE/i.test(cue)
      ? (cue.replace(/\/\s*10\b.*$/, '').match(/\d+(?:\.\d)?/g) ?? []).map(Number)
      : [],
  )
  const rpe = Math.max(...values)
  if (!(rpe > 0 && rpe <= 10)) return null
  return rpe >= 8 ? 'hard' : rpe >= 5 ? 'steady' : 'easy'
}

const effortOf = (text: string): SwimEffort | null =>
  rpeEffort(text) ?? EFFORTS.find(([, pattern]) => pattern.test(text))?.[0] ?? null

// "/" separates the parts of a split rep, except inside an effort cue such as "7/10".
const splitPieces = (text: string): string[] => {
  const pieces = ['']
  for (const segment of effortCueSegments(text)) {
    if (segment.cue) {
      pieces[pieces.length - 1] += segment.text
      continue
    }
    const [first, ...rest] = segment.text.split('/')
    pieces[pieces.length - 1] += first
    pieces.push(...rest)
  }
  return pieces.map(piece => piece.trim())
}

// A parenthesis may wrap onto the next lines; they join unless a blank line comes first.
const depth = (line: string): number =>
  (line.match(/\(/g)?.length ?? 0) - (line.match(/\)/g)?.length ?? 0)

const joinWrapped = (lines: string[]): string[] => {
  const joined: string[] = []
  for (let index = 0; index < lines.length; index++) {
    let line = lines[index]
    let open = depth(line)
    let end = index
    while (open > 0 && lines[end + 1]?.trim()) {
      end++
      line += ` ${lines[end]}`
      open += depth(lines[end])
    }
    if (open <= 0 && end > index) {
      joined.push(line)
      index = end
    } else joined.push(lines[index])
  }
  return joined
}

const clean = (text: string): string =>
  text
    .replace(/\s+/g, ' ')
    .replace(/^[\s,.:;-]+|[\s,.:;-]+$/g, '')
    .trim()

// A pool distance needs a unit unless it is a whole number of 25 m lengths.
type Quantity = Pick<SwimPart, 'meters' | 'seconds'>

const quantity = (value: string, unit: string | undefined): Quantity | null => {
  const amount = Number(value)
  if (!(amount > 0)) return null
  if (unit && !/^m$/i.test(unit)) return { meters: null, seconds: seconds(amount, unit) }
  if (!unit && amount % 25 !== 0) return null
  return { meters: amount, seconds: null }
}

const extractRest = (text: string): { rest: SwimRest | null; text: string } => {
  for (const [pattern, interval] of REST_TEXT) {
    const match = pattern.exec(text)
    if (match)
      return {
        rest: { min: seconds(Number(match[1]), match[2]), max: null, interval },
        text: text.replace(pattern, ' '),
      }
  }
  return { rest: null, text }
}

// The parts of a split rep ("25m drill/25m swim"). With `remainder`, a first part without a
// distance ("catch drill / 25 swim") takes the lengths left in the rep. A timed rep keeps its
// time: "2min hard/30sec float" is a hard piece and then a float.
const splitParts = (
  text: string,
  main: Quantity,
  unit: string | undefined,
  remainder: boolean,
): SwimPart[] | null => {
  const pieces = splitPieces(text)
  if (pieces.length < 2) return null
  const parsed = pieces.map(piece => {
    const match = PIECE.exec(piece)
    const part = match && quantity(match[1], match[2] ?? unit)
    return part && match ? { ...part, effort: effortOf(match[3]) } : null
  })
  const split = parsed.slice(1)
  if (!split.every(part => part !== null)) return null
  const [head] = parsed
  if (head) return [head, ...split]
  if (!remainder) return null
  const rest = (main.meters ?? 0) - split.reduce((sum, part) => sum + (part.meters ?? 0), 0)
  const lengths = split.every(part => part.meters !== null) && rest > 0
  return [{ ...main, ...(lengths && { meters: rest }), effort: effortOf(pieces[0]) }, ...split]
}

const amountLabel = (amount: string, unit: string | undefined): string =>
  !unit || /^m$/i.test(unit) ? `${amount}m` : `${amount} ${/^min/i.test(unit) ? 'min' : 's'}`

const repLabel = (reps: number | string, amount: string): string =>
  reps === 1 ? amount : `${reps} × ${amount}`

const parseStep = (line: string): SwimStep | null => {
  const match = STEP.exec(line)
  if (!match) return null
  const [, ladder, amount, unit, tail] = match
  const main = quantity(amount, unit)
  if (!main) return null
  // A ladder such as "16-12-8-4 x 25m" swims every listed count.
  const reps = ladder ? ladder.split('-').reduce((sum, count) => sum + Number(count), 0) : 1
  if (!(reps > 0)) return null
  const { rest, text } = extractRest(tail)
  const notes: string[] = []
  const descriptor = clean(
    text.replace(/\(([^)]*)\)/g, (_, note: string) => {
      if (clean(note)) notes.push(clean(note))
      return ' '
    }),
  ).replace(/^as\b\s*/i, '')
  const parts = splitParts(descriptor, main, unit, true) ?? [
    { ...main, effort: effortOf(descriptor) },
  ]
  if (descriptor) notes.unshift(descriptor)
  return { label: repLabel(ladder ?? reps, amountLabel(amount, unit)), reps, parts, rest, notes }
}

// "20 x 50m as" followed by "1 HARD (5sec), 1 EASY (20sec), repeat" alternates the reps.
const alternate = (block: SwimBlock | undefined, line: string): SwimBlock | null => {
  const match = ALTERNATION.exec(line)
  if (!match || block?.kind !== 'set' || block.repeat !== 1 || block.steps.length !== 1) return null
  const [step] = block.steps
  if (step.parts.length !== 1 || step.parts[0].effort !== null) return null
  const [
    ,
    firstCount,
    firstText,
    firstRest,
    firstUnit,
    secondCount,
    secondText,
    secondRest,
    secondUnit,
  ] = match
  const counts = [Number(firstCount), Number(secondCount)]
  const cycle = counts[0] + counts[1]
  if (!(cycle > 0) || step.reps % cycle !== 0) return null
  const [part] = step.parts
  const leg = (
    count: number,
    text: string,
    rest: string | undefined,
    unit: string | undefined,
  ): SwimStep => ({
    label: repLabel(count, step.label.replace(/^.*×\s*/, '')),
    reps: count,
    parts: [{ ...part, effort: effortOf(text) }],
    rest: rest ? { min: seconds(Number(rest), unit), max: null, interval: false } : null,
    notes: [text.toLowerCase()],
  })
  return {
    kind: 'set',
    repeat: step.reps / cycle,
    rest: null,
    steps: [
      leg(counts[0], firstText, firstRest, firstUnit),
      leg(counts[1], secondText, secondRest, secondUnit),
    ],
  }
}

export function parseSwimDescription(description: string): SwimStructure | null {
  const blocks: SwimBlock[] = []
  const lastStep = (): SwimStep | undefined => {
    const block = blocks.at(-1)
    return block?.kind === 'set' ? block.steps.at(-1) : undefined
  }
  for (const raw of joinWrapped(description.split(/\r?\n/))) {
    const line = clean(raw.replace(/https?:\/\/\S+/g, ''))
    if (!line) continue
    const roundRest = ROUND_REST.exec(line)
    const previous = blocks.at(-1)
    if (roundRest && previous?.kind === 'set' && previous.repeat > 1) {
      previous.rest = {
        min: seconds(Number(roundRest[1]), roundRest[2]),
        max: null,
        interval: false,
      }
      continue
    }
    const restLine = REST_LINE.exec(line) ?? roundRest
    if (restLine) {
      const unit = restLine === roundRest ? restLine[2] : restLine[3]
      const max = restLine === roundRest || !restLine[2] ? null : seconds(Number(restLine[2]), unit)
      blocks.push({
        kind: 'rest',
        rest: { min: seconds(Number(restLine[1]), unit), max, interval: false },
      })
      continue
    }
    const nested = NESTED.exec(line)
    const inner = nested && parseStep(nested[2])
    if (nested && inner && Number(nested[1]) > 0) {
      blocks.push({ kind: 'set', repeat: Number(nested[1]), steps: [inner], rest: null })
      continue
    }
    const bare = NESTED_BARE.exec(line)
    const bareInner = bare && parseStep(`${bare[2]} ${bare[3]}`)
    if (bare && bareInner && Number(bare[1]) > 0) {
      blocks.push({ kind: 'set', repeat: Number(bare[1]), steps: [bareInner], rest: null })
      continue
    }
    const step = parseStep(line)
    if (step) {
      blocks.push({ kind: 'set', repeat: 1, steps: [step], rest: null })
      continue
    }
    const alternated = alternate(previous, line)
    if (alternated) {
      blocks[blocks.length - 1] = alternated
      continue
    }
    // Any other line explains the step above it, and may carry that step's rest.
    const target = lastStep()
    if (!target) continue
    const { rest, text } = extractRest(line)
    if (rest && !target.rest) target.rest = rest
    const note = clean(clean(text).replace(/^\((.*)\)$/, '$1'))
    if (!note) continue
    target.notes.push(note)
    // "4 x 50" then "as (25 drill/25 swim)" splits the reps when the parts add up to one rep.
    const [only] = target.parts
    const split =
      target.parts.length === 1 && only.effort === null && only.meters !== null
        ? splitParts(note, only, undefined, false)
        : null
    if (split && split.reduce((sum, part) => sum + (part.meters ?? 0), 0) === only.meters)
      target.parts = split
  }

  const parts = blocks.flatMap(block =>
    block.kind === 'set' ? block.steps.flatMap(step => step.parts) : [],
  )
  if (parts.length === 0) return null
  const total = (key: 'meters' | 'seconds'): number =>
    blocks.reduce(
      (sum, block) =>
        block.kind === 'set'
          ? sum +
            block.repeat *
              block.steps.reduce(
                (stepSum, step) =>
                  stepSum +
                  step.reps * step.parts.reduce((partSum, part) => partSum + (part[key] ?? 0), 0),
                0,
              )
          : sum,
      0,
    )
  return {
    blocks,
    meters: total('meters'),
    seconds: total('seconds'),
    unit: parts.every(part => part.meters !== null)
      ? 'meters'
      : parts.every(part => part.seconds !== null)
        ? 'seconds'
        : null,
  }
}
