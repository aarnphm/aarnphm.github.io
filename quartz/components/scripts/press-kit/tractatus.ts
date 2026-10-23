import { runsFromRange, runsFromText, runsToPlain, runsToText, type Run } from '../press-kit-text'

export interface TractatusPassage {
  number: string
  depth: number
  runs: Run[]
}

export function selectedPropositions(range: Range, source: Element | null): TractatusPassage[] {
  if (!source) return []
  return Array.from(source.querySelectorAll('.tractatus-row')).flatMap(row => {
    const text = row.querySelector(':scope > .tractatus-text')
    const number = row.querySelector(':scope > .tractatus-number')?.textContent?.trim()
    if (!text || !number || !range.intersectsNode(text)) return []
    const clipped = document.createRange()
    clipped.selectNodeContents(text)
    if (clipped.compareBoundaryPoints(Range.START_TO_START, range) < 0)
      clipped.setStart(range.startContainer, range.startOffset)
    if (clipped.compareBoundaryPoints(Range.END_TO_END, range) > 0)
      clipped.setEnd(range.endContainer, range.endOffset)
    const runs = runsFromRange(clipped, source)
    if (!runsToPlain(runs).trim()) return []
    const depth = Number(row.parentElement?.getAttribute('data-tractatus-depth') ?? 0)
    return [{ number, depth: Number.isFinite(depth) ? depth : 0, runs }]
  })
}

export function propositionsText(passages: TractatusPassage[]): string {
  return passages.map(passage => `${passage.number} ${runsToText(passage.runs)}`).join('\n\n')
}

export function editedPropositions(text: string, original: TractatusPassage[]): TractatusPassage[] {
  const blocks = text.trim().split(/\n\s*\n/)
  const passages: TractatusPassage[] = []
  for (const block of blocks) {
    const match = /^(\d+(?:\.\d+)*)\s+([\s\S]+)$/.exec(block.trim())
    if (!match) return []
    const previous = original.find(passage => passage.number === match[1])
    if (!previous) return []
    passages.push({ ...previous, runs: runsFromText(match[2]) })
  }
  return passages
}
