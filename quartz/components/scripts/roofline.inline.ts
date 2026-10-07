type Stop = { b: number; x: number; y: number; p: string; pct: string; bound: string }

const setupRoofline = () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-roofline]')) {
    if (root.dataset.rflBound === 'true') continue
    const slider = root.querySelector<HTMLInputElement>('[data-rfl-batch]')
    const point = root.querySelector<SVGCircleElement>('[data-rfl-point]')
    const label = root.querySelector<SVGTextElement>('[data-rfl-point-label]')
    const canvas = root.querySelector<SVGElement>('[data-rfl-canvas]')
    if (!slider || !point || !label) continue
    root.dataset.rflBound = 'true'

    // The server computes every stop, so the script only moves marks and swaps text.
    const stops = JSON.parse(root.dataset.stops ?? '[]') as Stop[]
    const ridge = Math.round(Number(root.dataset.ridge))
    const set = (key: string, text: string) => {
      const el = root.querySelector<HTMLElement>(`[data-rfl-${key}]`)
      if (el) el.textContent = text
    }

    const handleInput = () => {
      const s = stops[Number(slider.value)]
      if (!s) return
      point.setAttribute('cx', String(s.x))
      point.setAttribute('cy', String(s.y))
      // Near the ridge the label would run under the prefill mark, so it flips to the left of the point.
      const flip = s.x > 360
      label.setAttribute('x', String(flip ? s.x - 9 : s.x + 9))
      label.setAttribute('y', String(s.y + 18))
      label.setAttribute('text-anchor', flip ? 'end' : 'start')
      label.textContent = `decode, batch ${s.b}`
      root.dataset.rflRegime = s.bound
      set('b', String(s.b))
      set('i', String(s.b))
      set('p', s.p)
      set('pct', `${s.pct}%`)
      set('state', s.bound)
      slider.setAttribute('aria-valuenow', String(s.b))
      slider.setAttribute('aria-valuetext', `batch ${s.b}, ${s.bound}-bound, ${s.p} TFLOP/s`)
      canvas?.setAttribute(
        'aria-label',
        `Roofline chart. Decode at batch ${s.b} has intensity ${s.b} FLOP/byte and is ${s.bound}-bound at ${s.p} TFLOP/s; the ridge is at ${ridge} FLOP/byte.`,
      )
    }

    slider.addEventListener('input', handleInput)
    handleInput()

    window.addCleanup(() => {
      slider.removeEventListener('input', handleInput)
      delete root.dataset.rflBound
    })
  }
}

document.addEventListener('nav', setupRoofline)
