import type { Runtime } from './runtime'

// the probe's one-line labels that an ellipsis can cut: feature names, chips, the card title,
// the titles under "in text", steering pins, a feature card's description
const CUT =
  '.lat-feat-chip, .lat-feat-head, .lat-feat-name, .lat-feat-cross a, .lat-feat-pin > span, .lat-chip, .lat-feat-about'

/**
 * Hovering a label that is cut short scrolls it to its end and back at a reading pace (the
 * status line's lat-marquee keyframes), so the whole name reads without a tooltip. The text is
 * wrapped in a span the first time, since a transform needs a box narrower than the clip.
 */
export function bindMarquee(rt: Runtime, root: HTMLElement): void {
  root.addEventListener(
    'pointerover',
    event => {
      const el = (event.target as Element | null)?.closest?.<HTMLElement>(CUT)
      if (!el || !root.contains(el) || el.dataset.marquee !== undefined) return
      if (el.scrollWidth <= el.clientWidth + 1) return
      let text = el.querySelector<HTMLElement>(':scope > .lat-marquee-text')
      if (!text) {
        text = document.createElement('span')
        text.className = 'lat-marquee-text'
        text.append(...el.childNodes)
        el.append(text)
      }
      const style = getComputedStyle(el)
      const room = el.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
      const shift = text.getBoundingClientRect().width - room
      if (shift <= 1) return
      el.style.setProperty('--lat-shift', `${-Math.ceil(shift)}px`)
      // about 40px a second, plus the holds at either end (a fifth of the cycle each); a short
      // overflow still takes 2.5 s, so a few cut letters do not flicker
      el.style.setProperty('--lat-marquee', `${Math.max(2.5, shift / 40 / 0.6).toFixed(2)}s`)
      el.dataset.marquee = ''
      el.addEventListener('pointerleave', () => delete el.dataset.marquee, { once: true })
    },
    { signal: rt.signal, passive: true },
  )
}
