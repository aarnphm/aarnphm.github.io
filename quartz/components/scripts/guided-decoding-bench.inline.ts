const setupGuidedDecodingBench = () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-guided-decoding-bench]')) {
    const tabs = [...root.querySelectorAll<HTMLButtonElement>('[data-gdb-tab]')]
    const panels = [...root.querySelectorAll<HTMLElement>('[data-gdb-panel]')]
    if (tabs.length === 0 || tabs.length !== panels.length) continue

    const select = (next: number, focus: boolean) => {
      tabs.forEach((tab, k) => {
        const on = k === next
        tab.setAttribute('aria-selected', on ? 'true' : 'false')
        tab.tabIndex = on ? 0 : -1
        panels[k].hidden = !on
      })
      if (focus) tabs[next].focus()
    }

    const onClick = (e: MouseEvent) => {
      const k = tabs.indexOf(e.currentTarget as HTMLButtonElement)
      if (k >= 0) select(k, false)
    }

    const onKey = (e: KeyboardEvent) => {
      const k = tabs.indexOf(e.currentTarget as HTMLButtonElement)
      const last = tabs.length - 1
      const next =
        e.key === 'ArrowRight'
          ? (k + 1) % tabs.length
          : e.key === 'ArrowLeft'
            ? (k + last) % tabs.length
            : e.key === 'Home'
              ? 0
              : e.key === 'End'
                ? last
                : -1
      if (next < 0) return
      e.preventDefault()
      select(next, true)
    }

    // Pointing at a QPS group or table row lights its partner, so a bar and its exact value read together.
    const onOver = (e: PointerEvent) => {
      const hit = (e.target as Element).closest<HTMLElement | SVGElement>('[data-gdb-q]')
      const panel = (e.target as Element).closest<HTMLElement>('[data-gdb-panel]')
      if (!panel) return
      if (hit) panel.dataset.gdbActive = hit.dataset.gdbQ
      else delete panel.dataset.gdbActive
    }
    const onLeave = () => {
      for (const panel of panels) delete panel.dataset.gdbActive
    }
    const fine = window.matchMedia('(hover: hover) and (pointer: fine)').matches

    for (const tab of tabs) {
      tab.addEventListener('click', onClick)
      tab.addEventListener('keydown', onKey)
    }
    if (fine) {
      root.addEventListener('pointerover', onOver)
      root.addEventListener('pointerleave', onLeave)
    }

    window.addCleanup(() => {
      for (const tab of tabs) {
        tab.removeEventListener('click', onClick)
        tab.removeEventListener('keydown', onKey)
      }
      root.removeEventListener('pointerover', onOver)
      root.removeEventListener('pointerleave', onLeave)
    })
  }
}

document.addEventListener('nav', setupGuidedDecodingBench)
