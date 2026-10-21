import { lidRibs } from '../../../util/base/vault/layout'
const SVG_NS = 'http://www.w3.org/2000/svg'

/** Two lines of at most `max` characters for the round lid label. */
function lidLines(title: string, max = 12): string[] {
  const lines = ['']
  for (const word of title.toUpperCase().split(/\s+/)) {
    const line = lines[lines.length - 1]
    if (!line || line.length + word.length + 1 <= max)
      lines[lines.length - 1] = line ? `${line} ${word}` : word
    else if (lines.length < 2) lines.push(word)
    else {
      lines[1] = `${lines[1]}…`
      break
    }
  }
  return lines.map(line => (line.length > max + 1 ? `${line.slice(0, max - 1)}…` : line))
}

/** The moulded lid relief and the round lid label, laid on the pulled can's lid. */
export function lidDetail(can: SVGGElement, title: string): SVGGElement | null {
  const lid = can.querySelector('.cn-lid')
  const transform = lid?.getAttribute('transform')
  const r = Number(lid?.getAttribute('r'))
  if (!transform || !Number.isFinite(r)) return null
  const node = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string>) => {
    const element = document.createElementNS(SVG_NS, tag)
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value)
    return element
  }
  const group = node('g', { class: 'cn-lid-detail', transform })
  const size = Math.round(r * 0.075 * 10) / 10
  const text = node('text', { class: 'cn-lid-text', 'font-size': String(size) })
  const lines = lidLines(title)
  lines.forEach((line, i) => {
    const span = node('tspan', {
      x: '0',
      y: String((i - (lines.length - 1) / 2) * size * 1.25 + size * 0.35),
    })
    span.textContent = line
    text.append(span)
  })
  const relief = node('g', { class: 'cn-rings' })
  relief.append(
    node('circle', { class: 'cn-lid-ring', r: String(Math.round(r * 9) / 10) }),
    node('path', { class: 'cn-lid-ribs', d: lidRibs(r) }),
  )
  group.append(
    relief,
    node('circle', { class: 'cn-lid-label', r: String(Math.round(r * 0.28)) }),
    text,
  )
  return group
}
