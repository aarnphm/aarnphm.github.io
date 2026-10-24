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

/** Measure cut boundaries once while the full bodies share a clean layout. */
function measureCuts(body: HTMLElement): (keep: number) => Cut {
  const top = body.getBoundingClientRect().top
  const ends = [...body.querySelectorAll<HTMLElement>('.fr-gloss, .fr-note')].map(
    el => el.getBoundingClientRect().bottom - top,
  )
  const range = document.createRange()
  range.selectNodeContents(body)
  const lines = [...range.getClientRects()].map(r => [r.top - top, r.bottom - top])
  return keep => {
    const fit = ends.filter(end => end <= keep)
    if (fit.length > 0) return { cut: Math.max(...fit), hidden: ends.length - fit.length }
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
}

export interface ClampFit {
  height: number
  /** Label centres relative to the group's top after fitting. */
  heads: number[]
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
     * Measure all groups together. Fitting then uses flex-column height deltas, so the ordered
     * collision pass can decide each following group's room without forcing layout per group.
     */
    measure(groups: HTMLElement[]) {
      const candidates = new Map<HTMLElement, { line: number; full: number }>()
      for (const group of groups) {
        for (const entry of group.querySelectorAll<HTMLElement>(':scope > .fr-entry')) {
          if (open.has(entry) || entry.querySelector('[data-fr-rev], [data-fr-notes]')) continue
          const body = bodyOf(entry)
          const line = parseFloat(getComputedStyle(body).lineHeight) || 18
          const full = body.offsetHeight
          if (full > (MIN_LINES + 2) * line) candidates.set(entry, { line, full })
        }
      }
      // Measure the controls in one write/read phase; hiding one removes its height and flex gap.
      for (const [entry] of controls) {
        if (open.has(entry)) show(entry, 'less')
        else if (candidates.has(entry)) show(entry, 'more')
      }
      return new Map(
        groups.map(group => {
          const rect = group.getBoundingClientRect()
          const entries = [...group.querySelectorAll<HTMLElement>(':scope > .fr-entry')].map(
            entry => {
              const body = bodyOf(entry)
              const button = controls.get(entry)!
              const label = entry.querySelector<HTMLElement>('.fr-entry-label')!
              const labelBox = label.getBoundingClientRect()
              const candidate = candidates.get(entry)
              return {
                entry,
                body,
                button,
                candidate,
                bodyHeight: body.getBoundingClientRect().height,
                controlSpace: button.hidden
                  ? 0
                  : button.getBoundingClientRect().height +
                    (parseFloat(getComputedStyle(entry).rowGap) || 0),
                head: labelBox.top + labelBox.height / 2 - rect.top,
                snap: candidate ? measureCuts(body) : null,
                count: body.querySelectorAll('.fr-gloss, .fr-note').length,
              }
            },
          )
          return [
            group,
            (room: number): ClampFit => {
              const states: {
                metric: (typeof entries)[number]
                shown: boolean
                cut: Cut | undefined
              }[] = entries.map(metric => ({
                metric,
                shown: open.has(metric.entry),
                cut: undefined,
              }))
              const delta = (state: (typeof states)[number]) =>
                (state.shown ? 0 : -state.metric.controlSpace) +
                (state.cut
                  ? Math.min(state.metric.bodyHeight, Math.floor(state.cut.cut)) -
                    state.metric.bodyHeight
                  : 0)
              const height = () =>
                rect.height + states.reduce((sum, state) => sum + delta(state), 0)
              const cuttable = states
                .filter(state => state.metric.candidate !== undefined)
                .sort((a, b) => a.metric.candidate!.full - b.metric.candidate!.full)
              if (height() > room) {
                for (const state of cuttable) state.shown = true
                let left =
                  cuttable.reduce((sum, state) => sum + state.metric.candidate!.full, 0) -
                  (height() - room)
                for (const [i, state] of cuttable.entries()) {
                  const { line, full } = state.metric.candidate!
                  const keep = Math.max(
                    MIN_LINES * line,
                    Math.min(full, left / (cuttable.length - i)),
                  )
                  const cut =
                    keep < full - 2 * line ? state.metric.snap!(keep) : { cut: 0, hidden: 0 }
                  if (cut.cut <= 0) {
                    state.shown = false
                    left -= full
                  } else {
                    state.cut = cut
                    left -= cut.cut
                  }
                }
                if (height() > room) {
                  for (const state of cuttable) {
                    if (state.cut) state.cut = { cut: 0, hidden: state.metric.count }
                  }
                }
              }
              let offset = 0
              const heads = states.map(state => {
                const head = state.metric.head + offset
                offset += delta(state)
                const { entry, body, button } = state.metric
                button.hidden = !state.shown
                if (state.cut) {
                  const { cut, hidden } = state.cut
                  body.setAttribute('data-fr-cap', '')
                  body.style.setProperty('--fr-cap', `${Math.floor(cut)}px`)
                  entry.setAttribute('data-fr-clamp', '')
                  show(
                    entry,
                    cut === 0
                      ? hidden > 0
                        ? `show ${hidden}`
                        : 'show'
                      : hidden > 0
                        ? `+ ${hidden} more`
                        : '… more',
                  )
                }
                return head
              })
              return { height: rect.height + offset, heads }
            },
          ]
        }),
      )
    },
  }
}
