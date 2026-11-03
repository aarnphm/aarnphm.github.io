import type { mountClamps } from './fr-margins-clamp'

interface InteractionOptions {
  article: HTMLElement
  signal: AbortSignal
  groups: HTMLElement[]
  anchors: Map<HTMLElement, HTMLElement>
  bodies: Map<HTMLElement, HTMLElement>
  notes: HTMLElement[]
  leaderOf: Map<HTMLElement, SVGPathElement[]>
  clamps: ReturnType<typeof mountClamps>
}

export function mountInteractions({
  article,
  signal,
  groups,
  anchors,
  bodies,
  notes,
  leaderOf,
  clamps,
}: InteractionOptions) {
  const settle = (entry: HTMLElement) => {
    const body = bodies.get(entry)!
    const group = entry.parentElement as HTMLElement
    const anchor = anchors.get(group)!
    const out =
      article.dataset.frMode === 'inline' &&
      entry.hasAttribute('data-open') &&
      group.parentElement === anchor
    if (out) {
      let fold = anchor.querySelector<HTMLElement>(':scope > .fr-fold')
      if (!fold) {
        fold = document.createElement('span')
        fold.className = 'fr-fold'
        anchor.append(fold)
      }
      if (body.parentElement !== fold) fold.append(body)
    } else {
      if (body.parentElement !== entry) entry.append(body)
      const fold = anchor.querySelector(':scope > .fr-fold')
      if (fold && fold.childElementCount === 0) fold.remove()
    }
  }

  for (const label of article.querySelectorAll<HTMLButtonElement>('.fr-entry-label')) {
    label.addEventListener(
      'click',
      () => {
        if (article.dataset.frMode !== 'inline') return
        const entry = label.closest<HTMLElement>('.fr-entry')
        if (!entry) return
        const open = entry.toggleAttribute('data-open')
        label.setAttribute('aria-expanded', String(open))
        settle(entry)
      },
      { signal },
    )
  }

  const labelOf = (content: HTMLElement) =>
    content.previousElementSibling?.querySelector<HTMLElement>('.sidenote-label') ?? null
  const siblings = new Map<HTMLElement, HTMLElement[]>()
  const lights = new Map<HTMLElement, (on: boolean) => void>()
  for (const group of groups) {
    const anchor = anchors.get(group)!
    siblings.set(anchor, [...(siblings.get(anchor) ?? []), group])
  }
  for (const [anchor, mine] of siblings) {
    const light = (on: boolean) => {
      anchor.classList.toggle('fr-hot', on)
      for (const group of mine) {
        group.classList.toggle('fr-hot', on)
        for (const path of leaderOf.get(group) ?? []) path.classList.toggle('fr-hot', on)
      }
    }
    for (const group of mine) lights.set(group, light)
    for (const node of [anchor, ...mine]) {
      node.addEventListener('pointerenter', () => light(true), { signal })
      node.addEventListener('pointerleave', () => light(false), { signal })
    }
  }
  for (const content of notes) {
    const light = (on: boolean) => {
      for (const path of leaderOf.get(content) ?? []) path.classList.toggle('fr-hot', on)
    }
    for (const node of [content, labelOf(content)]) {
      node?.addEventListener('pointerenter', () => light(true), { signal })
      node?.addEventListener('pointerleave', () => light(false), { signal })
    }
  }

  for (const cite of article.querySelectorAll<HTMLAnchorElement>('a[data-bib]')) {
    const id = decodeURIComponent(new URL(cite.href).hash.slice(1))
    const entry = id
      ? article
          .querySelector<HTMLElement>(`.fr-gloss#${CSS.escape(id)}`)
          ?.closest<HTMLElement>('.fr-entry')
      : null
    if (!entry) continue
    const light = lights.get(entry.parentElement as HTMLElement)
    let lit = false
    let held = false
    const release = (event: PointerEvent) => {
      const to = event.relatedTarget as Node | null
      if (!lit || (to && (cite.contains(to) || entry.contains(to)))) return
      lit = false
      entry.classList.remove('fr-hot')
      light?.(false)
      if (held) clamps.hold(entry, false)
      held = false
    }
    cite.addEventListener(
      'pointerenter',
      () => {
        if (article.dataset.frMode !== 'margin') return
        light?.(true)
        if (lit) return
        lit = true
        entry.classList.add('fr-hot')
        held = clamps.hold(entry, true)
      },
      { signal },
    )
    cite.addEventListener('pointerleave', release, { signal })
    entry.addEventListener('pointerleave', release, { signal })
  }
  return settle
}
