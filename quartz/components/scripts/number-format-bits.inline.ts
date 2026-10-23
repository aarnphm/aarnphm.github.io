import { registerMdxContent } from './mdx-content'

type ValueState = {
  bits: Record<string, string>
  dec: Record<string, string>
  detail: Record<string, Record<string, string>>
}

const setupNumberFormatBits = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-nfb-states')
  if (!raw) return null
  const states = JSON.parse(raw) as Record<string, ValueState>
  const rows = Array.from(root.querySelectorAll<HTMLElement>('[data-nfb-row]'))
  const outputs = Array.from(root.querySelectorAll<HTMLElement>('[data-nfb-out]'))

  const checked = (selector: string) =>
    root.querySelector<HTMLInputElement>(`${selector}:checked`)?.value

  const apply = () => {
    const value = checked('[data-nfb-value-input]')
    const format = checked('[data-nfb-format-input]')
    const state = value ? states[value] : undefined
    if (!state || !format) return
    for (const row of rows) {
      const id = row.getAttribute('data-nfb-row') ?? ''
      const bits = state.bits[id] ?? ''
      row
        .querySelectorAll('.nfb-bit')
        .forEach((cell, i) => cell.setAttribute('data-nfb-on', bits[i]))
      const dec = row.querySelector('[data-nfb-dec]')
      if (dec) dec.textContent = state.dec[id]
    }
    // The detail strings are KaTeX markup rendered at build time from the figure's own model.
    const detail = state.detail[format]
    for (const out of outputs) out.innerHTML = detail[out.getAttribute('data-nfb-out') ?? ''] ?? ''
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent('[data-number-format-bits]', root => {
  const cleanup = setupNumberFormatBits(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
