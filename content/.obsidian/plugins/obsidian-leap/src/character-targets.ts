import type { Text } from '@codemirror/state'
import type { EditorView } from '@codemirror/view'
import type { LeapTarget } from './types'

const equivalenceClasses = [' \t\r\n', '([{', ')]}', '\'"`']
const safeLabels = 'sfnut/SFNLHMUGTZ?'

function characterAt(doc: Text, position: number): string {
  const point = doc.sliceString(position, Math.min(position + 2, doc.length)).codePointAt(0)
  return point === undefined ? '' : String.fromCodePoint(point)
}

function previousPosition(doc: Text, position: number): number {
  const preceding = Array.from(doc.sliceString(Math.max(0, position - 2), position)).at(-1)
  return position - (preceding?.length ?? 0)
}

function isBoundary(doc: Text, position: number): boolean {
  if (position === 0 || position === doc.length) return true
  const units = doc.sliceString(position - 1, position + 1)
  const previous = units.charCodeAt(0)
  const current = units.charCodeAt(1)
  return !(previous >= 0xd800 && previous <= 0xdbff && current >= 0xdc00 && current <= 0xdfff)
}

export function characterTargets(
  view: EditorView,
  origin: number,
  character: string,
  backward: boolean,
  till: boolean,
): LeapTarget[] {
  if (!view.inView || Array.from(character).length !== 1) return []

  const equivalents = equivalenceClasses.find(group => group.includes(character)) ?? character
  const doc = view.state.doc
  const clip = view.scrollDOM.getBoundingClientRect()
  const targets: LeapTarget[] = []

  for (const range of view.visibleRanges) {
    for (let from = range.from; from <= range.to; ) {
      const virtualEnd = from === doc.length
      if (from === range.to && (!virtualEnd || !equivalents.includes('\n'))) break
      if (!isBoundary(doc, from)) {
        from++
        continue
      }

      const candidate = virtualEnd ? '\n' : characterAt(doc, from)
      const to = virtualEnd ? from : from + candidate.length
      const position = from
      from = virtualEnd ? from + 1 : to
      if (to > range.to || !equivalents.includes(candidate)) continue
      if (backward ? position >= origin : position <= origin) continue

      const line = doc.lineAt(position)
      const previous = position > line.from ? characterAt(doc, previousPosition(doc, position)) : ''
      const next = to < line.to ? characterAt(doc, to) : ''
      const previousMatches = previous !== '' && equivalents.includes(previous)
      const nextMatches = next !== '' && equivalents.includes(next)
      // Leap labels both ends of equivalent-character runs, and only adds an EOL
      // whitespace alias when trailing whitespace has not already supplied it.
      if (candidate === '\n' ? previousMatches : previousMatches && nextMatches) continue

      const rect =
        candidate === '\n' ? view.coordsAtPos(position, -1) : view.coordsForChar(position)
      if (
        !rect ||
        rect.bottom <= clip.top ||
        rect.top >= clip.bottom ||
        rect.right < clip.left ||
        rect.left >= clip.right
      ) {
        continue
      }

      let cursor = position
      if (till) {
        cursor = backward ? to : previousPosition(doc, position)
        if (!backward && position === line.from && position > 0) {
          const previousLine = doc.lineAt(position - 1)
          if (previousLine.length > 0) cursor = previousPosition(doc, cursor)
        }
      }
      if (cursor === origin) continue
      targets.push({ from: position, to, cursor })
    }
  }

  return backward ? targets.reverse() : targets
}

export function characterLabelAlphabet(motion: 'f' | 'F' | 't' | 'T'): string[] {
  const forward = motion.toLowerCase()
  const backward = forward.toUpperCase()
  return [
    forward,
    ...Array.from(safeLabels).filter(label => label !== forward && label !== backward),
  ]
}
