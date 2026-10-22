/** Fit the rack before shaping labels; yield background preparation whenever someone interacts. */
export function prepareVaultScene(root: HTMLElement) {
  const source = root.querySelector<HTMLTemplateElement>('template.vault-source')
  const svg = (source?.content ?? root).querySelector<SVGSVGElement>('svg.vault-scene')
  if (!svg) return null
  const status = root.querySelector<HTMLElement>('.vault-status')
  const statusText = status?.querySelector('.vault-status-text')
  const setStatus = (state: 'loading' | 'ready') => {
    if (status) status.dataset.state = state
    if (statusText) statusText.textContent = state === 'ready' ? 'vault ready' : 'preparing vault'
  }
  const pending = new Map<number, { can: SVGGElement; labels: SVGTextElement[] }>()
  if (source) {
    for (const can of Array.from(svg.querySelectorAll<SVGGElement>(':scope > .cn-can'))) {
      const labels = Array.from(can.querySelectorAll<SVGTextElement>('.cn-title, .cn-line'))
      for (const label of labels) label.style.display = 'none'
      pending.set(Number(can.dataset.index), { can, labels })
    }
  }
  let frame = 0
  let resumeTimer: number | null = null
  const win = root.ownerDocument.defaultView ?? window
  const ensure = (index: number) => {
    const entry = pending.get(index)
    if (!entry) return
    for (const label of entry.labels) label.style.display = ''
    pending.delete(index)
  }
  const start = (signal: AbortSignal) => {
    const matrix = svg.getScreenCTM()
    const viewport = root.querySelector('.vault-viewport')?.getBoundingClientRect()
    const inView = (can: SVGGElement) => {
      if (!matrix || !viewport) return false
      const point = new DOMPoint(Number(can.dataset.cx), Number(can.dataset.cy)).matrixTransform(
        matrix,
      )
      return (
        point.x >= viewport.left &&
        point.x <= viewport.right &&
        point.y >= viewport.top &&
        point.y <= viewport.bottom
      )
    }
    const indices = [...pending]
      .map(([index, { can }]) => ({ index, visible: inView(can) }))
      .sort((a, b) => Number(b.visible) - Number(a.visible) || a.index - b.index)
      .map(({ index }) => index)
    let cursor = 0
    let holding = false
    const batch = () => {
      frame = 0
      if (signal.aborted || holding) return
      const deadline = performance.now() + 4
      for (let count = 0; cursor < indices.length && count < 8; count++) {
        ensure(indices[cursor++])
        if (performance.now() >= deadline) break
      }
      if (pending.size) frame = requestAnimationFrame(batch)
      else {
        root.dataset.vaultReady = 'ready'
        root.removeAttribute('aria-busy')
        setStatus('ready')
      }
    }
    const pause = () => {
      cancelAnimationFrame(frame)
      frame = 0
      if (resumeTimer !== null) win.clearTimeout(resumeTimer)
      resumeTimer = win.setTimeout(() => {
        resumeTimer = null
        if (!signal.aborted && !holding && pending.size) frame = requestAnimationFrame(batch)
      }, 180)
    }
    root.addEventListener(
      'pointerdown',
      () => {
        holding = true
        pause()
      },
      { capture: true, signal },
    )
    const release = () => {
      holding = false
      pause()
    }
    win.addEventListener('pointerup', release, { signal })
    win.addEventListener('pointercancel', release, { signal })
    root.addEventListener('lostpointercapture', release, { signal })
    win.addEventListener('blur', release, { signal })
    root.addEventListener('wheel', pause, { passive: true, signal })
    root.addEventListener('keydown', pause, { signal })
    if (indices.length) {
      root.setAttribute('aria-busy', 'true')
      frame = requestAnimationFrame(() => {
        frame = requestAnimationFrame(batch)
      })
    } else batch()
  }
  const dispose = () => {
    cancelAnimationFrame(frame)
    if (resumeTimer !== null) win.clearTimeout(resumeTimer)
    for (const index of pending.keys()) ensure(index)
    if (source) source.content.append(svg)
    root.removeAttribute('aria-busy')
    setStatus('loading')
  }
  return { svg, ensure, start, dispose }
}

export type VaultScene = NonNullable<ReturnType<typeof prepareVaultScene>>
