const SVG = 'http://www.w3.org/2000/svg'

type Attributes = Record<string, string | number>

function shape(name: string, attributes: Attributes): SVGElement {
  const node = document.createElementNS(SVG, name)
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value))
  return node
}

function random(seed: string): () => number {
  let state = 2166136261
  for (const char of seed) state = Math.imul(state ^ char.charCodeAt(0), 16777619)
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

function path(parent: SVGElement, d: string, width = 0.8): void {
  parent.append(shape('path', { d, fill: 'none', stroke: 'currentColor', 'stroke-width': width }))
}

// The blade has a terminal lobe and two side lobes; veins meet its stalk at the base.
function leaf(parent: SVGElement, x: number, y: number, angle: number, scale: number): void {
  const group = shape('g', { transform: `translate(${x} ${y}) rotate(${angle}) scale(${scale})` })
  group.append(
    shape('path', {
      d: 'M0 0 C-3-1-7-1-9-6 L-5-6 C-9-9-10-13-8-16 L-4-13 C-5-18-2-21 0-25 C2-21 5-18 4-13 L8-16 C10-13 9-9 5-6 L9-6 C7-1 3-1 0 0Z',
      fill: 'var(--light)',
      stroke: 'currentColor',
      'stroke-width': 0.8,
    }),
  )
  path(
    group,
    'M0 0 Q1-10 0-22 M0-5 L-6-8 M0-5 L6-8 M0-10 L-6-14 M0-10 L6-14 M0-14 L-2-18 M0-14 L2-18',
    0.55,
  )
  parent.append(group)
}

function grapes(parent: SVGElement, x: number, y: number, scale: number): void {
  const group = shape('g', { transform: `translate(${x} ${y}) scale(${scale})` })
  path(group, 'M0-7 Q-2-3 0 0 M0-3 Q4-7 7-5', 0.7)
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4 - row; col++) {
      const cx = (col - (3 - row) / 2) * 4.2
      const cy = row * 3.7
      group.append(
        shape('circle', {
          cx,
          cy,
          r: 2.45,
          fill: 'var(--light)',
          stroke: 'currentColor',
          'stroke-width': 0.65,
        }),
      )
      path(group, `M${cx - 1.2} ${cy + 0.2} q0 1.1 1.3 1.3`, 0.45)
    }
  }
  parent.append(group)
}

function tendril(parent: SVGElement, x: number, y: number, direction: number, size: number): void {
  let d = `M${x} ${y}`
  for (let i = 1; i <= 44; i++) {
    const t = i / 44
    const angle = t * Math.PI * 3.4
    const radius = size * (1 - t * 0.85)
    d += ` L${(x + direction * (size - Math.cos(angle) * radius)).toFixed(2)} ${(y + Math.sin(angle) * radius).toFixed(2)}`
  }
  path(parent, d, 0.65)
}

function vine(parent: SVGElement, width: number, seed: string, compact = false): void {
  const next = random(seed)
  const step = compact ? 30 : 52
  const amplitude = compact ? 3 : 5
  const count = Math.max(2, Math.floor(width / step))
  const spacing = width / count
  const phase = next() * Math.PI
  const heightAt = (x: number) => Math.sin((x / spacing) * Math.PI + phase) * amplitude
  let stem = `M0 ${heightAt(0).toFixed(2)}`
  for (let x = 2; x <= width; x += 2) stem += ` L${x} ${heightAt(x).toFixed(2)}`
  path(parent, stem, compact ? 0.9 : 1.1)
  for (let i = 0; i < count; i++) {
    const x = (i + 0.35 + next() * 0.3) * spacing
    const y = heightAt(x)
    const side = i % 2 === 0 ? -1 : 1
    const reach = (compact ? 4 : 6) + next() * 4
    const tipX = x + 8 + next() * 5
    const tipY = y + side * reach
    path(parent, `M${x} ${y} Q${x + 4} ${tipY} ${tipX} ${tipY}`, 0.7)
    leaf(
      parent,
      tipX,
      tipY,
      side < 0 ? 25 + next() * 45 : 150 + next() * 40,
      compact ? 0.48 + next() * 0.15 : 0.58 + next() * 0.2,
    )
    if (i % 3 === 1) {
      path(parent, `M${x} ${y} Q${x - 8} ${y + 4} ${x - 6} ${y + 10}`, 0.65)
      grapes(parent, x - 6, y + 13, compact ? 0.55 : 0.72)
    } else {
      tendril(parent, x - 3, y, -1, compact ? 4 : 6 + next() * 2)
    }
  }
}

function canvas(width: number, height: number): SVGElement {
  return shape('svg', {
    viewBox: `0 0 ${width} ${height}`,
    'aria-hidden': 'true',
    focusable: 'false',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
  })
}

function separator(width: number, seed: string): SVGElement {
  const svg = canvas(width, 64)
  const group = shape('g', { transform: 'translate(12 31)' })
  vine(group, width - 24, seed)
  svg.append(group)
  return svg
}

function initial(letter: string, seed: string): SVGElement {
  const svg = canvas(160, 160)
  svg.append(
    shape('rect', {
      x: 7,
      y: 7,
      width: 146,
      height: 146,
      fill: 'var(--light)',
      stroke: 'currentColor',
      'stroke-width': 1,
    }),
  )
  svg.append(
    shape('rect', {
      x: 11,
      y: 11,
      width: 138,
      height: 138,
      fill: 'none',
      stroke: 'currentColor',
      'stroke-width': 0.45,
    }),
  )
  const sides = [
    'translate(15 18)',
    'translate(142 15) rotate(90)',
    'translate(145 142) rotate(180)',
    'translate(18 145) rotate(270)',
  ]
  sides.forEach((transform, i) => {
    const group = shape('g', { transform })
    vine(group, 130, `${seed}:${i}`, true)
    svg.append(group)
  })
  const text = shape('text', {
    x: 80,
    y: 124,
    'text-anchor': 'middle',
    'font-size': 132,
    'font-family': 'var(--mag-display)',
    fill: 'currentColor',
    stroke: 'var(--light)',
    'stroke-width': 3,
    'paint-order': 'stroke',
  })
  text.textContent = letter
  svg.append(text)
  return svg
}

/** Seeded botanical decoration stays local and redraws only when its available width changes. */
export function mountProfileOrnaments(): () => void {
  const seed = document.body.dataset.slug ?? location.pathname
  const decorated: HTMLElement[] = []
  const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
      const width = Math.round(entry.contentRect.width)
      if (width < 40 || !(entry.target instanceof HTMLElement)) continue
      const host = entry.target
      if (host.dataset.width === String(width)) continue
      host.dataset.width = String(width)
      host.replaceChildren(separator(width, `${seed}:${host.dataset.chapter}`))
    }
  })
  document.querySelectorAll<HTMLElement>('article .mag-chapter-vine').forEach(host => {
    decorated.push(host)
    observer.observe(host)
  })
  const initials = [
    ...document.querySelectorAll<HTMLElement>('.mag-initial[data-face="procedural"]'),
  ]
  initials.forEach((host, index) => {
    host.append(initial(host.dataset.letter ?? '', `${seed}:initial:${index}`))
    host.classList.add('has-procedural')
  })
  return () => {
    observer.disconnect()
    decorated.forEach(node => {
      node.replaceChildren()
      delete node.dataset.width
    })
    initials.forEach(host => {
      host.querySelector('svg')?.remove()
      host.classList.remove('has-procedural')
    })
  }
}
