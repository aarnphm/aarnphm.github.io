// A margin group gets the room between its number and the next entry in the margin column; a body
// taller than that is cut at a whole reference or a whole line and closes with a `more` control, so
// one long bibliography or note cannot push every later group away from its number. The reader can
// open a cut body; an open body takes the height it needs and pushes the column down.

// a cut body keeps at least this many lines, so the entry still reads as its kind
const MIN_LINES = 3

interface Cut {
  /** the body's visible height */
  cut: number
  /** references or notes below the cut */
  hidden: number
}

/** Bottom of the last reference or line that ends above `keep`, relative to the body's top. */
function snap(body: HTMLElement, keep: number): Cut {
  const top = body.getBoundingClientRect().top
  const items = [...body.querySelectorAll<HTMLElement>('.fr-gloss, .fr-note')]
  const ends = items.map(el => el.getBoundingClientRect().bottom - top)
  const fit = ends.filter(end => end <= keep)
  if (fit.length > 0) return { cut: Math.max(...fit), hidden: ends.length - fit.length }
  // no whole reference fits: cut above the first line the limit crosses
  const range = document.createRange()
  range.selectNodeContents(body)
  const lines = [...range.getClientRects()].map(r => [r.top - top, r.bottom - top])
  let cut = keep
  for (let moved = true; moved; ) {
    moved = false
    for (const [t, b] of lines)
      if (t < cut && b > cut && t > 0) {
        cut = t
        moved = true
      }
  }
  return { cut, hidden: 0 }
}

export function mountClamps(article: HTMLElement, relayout: () => void, signal: AbortSignal) {
  const open = new Set<HTMLElement>()
  const controls = new Map<HTMLElement, HTMLButtonElement>()
  const bodyOf = (entry: HTMLElement) => entry.querySelector<HTMLElement>('.fr-entry-body')!
  for (const [i, entry] of [...article.querySelectorAll<HTMLElement>('.fr-entry')].entries()) {
    const body = bodyOf(entry)
    body.id ||= `fr-body-${entry.closest<HTMLElement>('.fr-group')?.dataset.n ?? 0}-${i}`
    const more = document.createElement('button')
    more.type = 'button'
    more.className = 'fr-btn fr-link fr-more'
    more.hidden = true
    more.setAttribute('aria-controls', body.id)
    more.addEventListener(
      'click',
      () => {
        if (open.has(entry)) open.delete(entry)
        else open.add(entry)
        relayout()
      },
      { signal },
    )
    // a link past the cut opens the body when it takes focus, so the keyboard never lands unseen
    body.addEventListener(
      'focusin',
      event => {
        if (!body.hasAttribute('data-fr-cap')) return
        const target = event.target as Element
        if (!target.matches(':focus-visible')) return
        if (target.getBoundingClientRect().bottom <= body.getBoundingClientRect().bottom) return
        open.add(entry)
        relayout()
      },
      { signal },
    )
    entry.append(more)
    controls.set(entry, more)
  }

  const show = (entry: HTMLElement, label: string) => {
    const more = controls.get(entry)!
    more.hidden = false
    more.textContent = label
    more.setAttribute('aria-expanded', String(open.has(entry)))
  }

  // hiding a focused control blurs it, so the control that had focus takes it back after the layout
  let focused: HTMLButtonElement | null = null
  return {
    /** Every body at its full height, controls hidden: the state the layout measures from. */
    reset() {
      const active = document.activeElement
      focused = [...controls.values()].find(more => more === active) ?? null
      for (const [entry, more] of controls) {
        const body = bodyOf(entry)
        body.removeAttribute('data-fr-cap')
        body.style.removeProperty('--fr-cap')
        entry.removeAttribute('data-fr-clamp')
        more.hidden = true
      }
    },
    /** Returns focus to its control once the layout has shown it again. */
    refocus() {
      if (focused && !focused.hidden) focused.focus({ preventScroll: true })
      focused = null
    },
    /**
     * Cuts the group's bodies until it is no taller than `room`: each keeps an even share of the
     * height left, a short body keeps all of itself, and no body drops under its minimum.
     */
    fit(group: HTMLElement, room: number) {
      const entries = [...group.querySelectorAll<HTMLElement>(':scope > .fr-entry')]
      // an opened body keeps its `less` control wherever it falls
      for (const entry of entries) if (open.has(entry)) show(entry, 'less')
      const height = () => group.getBoundingClientRect().height
      if (height() <= room) return
      const cuttable = entries
        // the review deck and the reader's notes are worked in place, so they keep their height
        .filter(entry => !open.has(entry) && !entry.querySelector('[data-fr-rev], [data-fr-notes]'))
        .map(entry => {
          const body = bodyOf(entry)
          const line = parseFloat(getComputedStyle(body).lineHeight) || 18
          return { entry, body, line, full: body.offsetHeight }
        })
        // a cut that saves less than the control's own line is no cut
        .filter(c => c.full > (MIN_LINES + 2) * c.line)
        .sort((p, q) => p.full - q.full)
      // the controls take their lines before the shares are counted
      for (const { entry } of cuttable) show(entry, 'more')
      let left = cuttable.reduce((sum, c) => sum + c.full, 0) - (height() - room)
      const cap = (entry: HTMLElement, body: HTMLElement, { cut, hidden }: Cut) => {
        body.setAttribute('data-fr-cap', '')
        body.style.setProperty('--fr-cap', `${Math.floor(cut)}px`)
        entry.setAttribute('data-fr-clamp', '')
        // a closed body counts what it holds; a cut one counts what is left past the cut
        if (cut === 0) show(entry, hidden > 0 ? `show ${hidden}` : 'show')
        else show(entry, hidden > 0 ? `+ ${hidden} more` : '… more')
      }
      for (const [i, { entry, body, line, full }] of cuttable.entries()) {
        const keep = Math.max(MIN_LINES * line, Math.min(full, left / (cuttable.length - i)))
        const cut = keep < full - 2 * line ? snap(body, keep) : { cut: 0, hidden: 0 }
        if (cut.cut <= 0) {
          controls.get(entry)!.hidden = true
          left -= full
          continue
        }
        cap(entry, body, cut)
        left -= cut.cut
      }
      // blocks set close together leave no room even for the minimum: the cut bodies close to
      // their label and control, which still name the kind and the number
      if (height() <= room) return
      for (const { entry, body } of cuttable)
        if (entry.hasAttribute('data-fr-clamp'))
          cap(entry, body, { cut: 0, hidden: body.querySelectorAll('.fr-gloss, .fr-note').length })
    },
  }
}
