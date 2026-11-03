import {
  clearCache,
  layoutNextLine,
  measureNaturalWidth,
  prepare,
  prepareWithSegments,
} from '@chenglou/pretext'

export function prepareLabel(text: string, font: string, letterSpacing: number) {
  const options = { letterSpacing }
  const prepared = prepareWithSegments(text, font, options)
  return {
    prepared,
    text,
    width: measureNaturalWidth(prepared),
    ellipsis: measureNaturalWidth(prepare('…', font, options)),
    space: measureNaturalWidth(prepare('\u00a0', font, options)),
    letterSpacing,
  }
}

type Label = ReturnType<typeof prepareLabel>

export function fitLabel(label: Label, maxWidth: number): string {
  if (maxWidth <= 0) return ''
  if (label.width <= maxWidth) return label.text
  if (label.ellipsis > maxWidth) return ''
  const room = maxWidth - label.ellipsis - label.letterSpacing
  let text = ''
  let cursor = { segmentIndex: 0, graphemeIndex: 0 }
  let width = 0
  // Continue past word boundaries to fill the last line up to a whole grapheme.
  for (;;) {
    const piece = layoutNextLine(label.prepared, room - width, cursor)
    if (!piece || piece.width > room - width) break
    width += piece.width
    if (piece.text.endsWith(' ')) {
      if (width + label.space > room) {
        text += piece.text.trimEnd()
        break
      }
      width += label.space
    }
    text += piece.text
    cursor = piece.end
  }
  return `${text.trimEnd()}…`
}

export function createGroupLabelFitter() {
  let elements = new WeakMap<
    SVGTextElement,
    {
      text: string
      family: string
      style: string
      weight: string
      letterSpacing: number
      sizes: Map<number, Label>
    }
  >()
  return {
    clear() {
      clearCache()
      elements = new WeakMap()
    },
    fit(element: SVGTextElement, text: string, size: number, width: number): string {
      let entry = elements.get(element)
      if (!entry || entry.text !== text) {
        const style = getComputedStyle(element)
        entry = {
          text,
          family: style.fontFamily,
          style: style.fontStyle,
          weight: style.fontWeight,
          letterSpacing: Number.parseFloat(style.letterSpacing) || 0,
          sizes: new Map(),
        }
        elements.set(element, entry)
        void document.fonts
          .load(`${entry.style} ${entry.weight} 24px ${entry.family}`, text)
          .catch(() => {})
      }
      let label = entry.sizes.get(size)
      if (!label) {
        const font = `${entry.style} ${entry.weight} ${size}px ${entry.family}`
        label = prepareLabel(text, font, entry.letterSpacing)
        // The camera's normal range has fewer than 32 quarter-octave sizes.
        if (entry.sizes.size >= 32) entry.sizes.delete(entry.sizes.keys().next().value ?? size)
        entry.sizes.set(size, label)
      }
      return fitLabel(label, width)
    },
  }
}
