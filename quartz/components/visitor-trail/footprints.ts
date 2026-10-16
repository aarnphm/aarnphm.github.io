export type FootprintKind = 'boot' | 'paw' | 'hoof' | 'blob'

const impressions: Record<FootprintKind, string> = {
  boot: 'M-3-1h1.5v2H-3ZM-1-1.5h2.5v.5h1v2h-1v.5H-1Z',
  paw: 'M-1-1.5Q-3-1.5-3 0T-1 1.5Q1 2 1 0T-1-1.5ZM1-2h1v1H1ZM2-.5h1v1H2ZM1 1h1v1H1Z',
  hoof: 'M-2-2h3v1h1v.5h-4ZM-2 .5h4V1H1v1h-3Z',
  blob: 'M-3 .5Q-3-1.5-1-1.5T1.5-.5Q3 1 1 1.5T-3 .5Z',
}

// `spread` is how far each print sits off the line of travel, in scene units.
export function footprintTransform(x: number, y: number, angle: number, side: number, spread = 2) {
  const offset = side * spread
  const px = x - Math.sin(angle) * offset
  const py = y + Math.cos(angle) * offset
  return `translate(${px.toFixed(1)} ${py.toFixed(1)}) rotate(${((angle * 180) / Math.PI).toFixed(1)}) scale(1 ${side})`
}

export function footprint(kind: FootprintKind, transform: string) {
  return `<path data-footprint="${kind}" d="${impressions[kind]}" transform="${transform}"/>`
}

export function impression(document: Document, kind: FootprintKind) {
  const mark = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  mark.setAttribute('d', impressions[kind])
  return mark
}

export const bootImpression = (document: Document) => impression(document, 'boot')
