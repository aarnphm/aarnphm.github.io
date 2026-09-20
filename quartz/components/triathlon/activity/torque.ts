export function setupTorqueCharts(scope: HTMLElement): () => void {
  const showCell = (graph: SVGSVGElement, index: number): void => {
    const cells = Array.from(graph.querySelectorAll<SVGRectElement>('[data-torque-cell]'))
    const selected = Math.max(0, Math.min(cells.length - 1, index))
    const cell = cells[selected]
    if (!cell) return
    const text = cell.dataset.torqueCell ?? ''
    for (const candidate of cells)
      candidate.classList.toggle('tri-torque-cell--active', candidate === cell)
    graph.setAttribute('aria-valuenow', String(selected))
    graph.setAttribute('aria-valuetext', text)
    const readout = graph.closest('.tri-torque-cadence')?.querySelector('.tri-torque-readout')
    if (readout) readout.textContent = text
  }
  const click = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('.tri-torque-mode')
    const panel = button?.closest<HTMLElement>('.tri-torque-panel')
    if (!button || !panel) return
    const mode = button.dataset.torqueMode === 'cadence' ? 'cadence' : 'distance'
    panel.dataset.torqueMode = mode
    panel.closest('.tri-act')?.classList.remove('tri-act--scrub')
    for (const pane of panel.querySelectorAll<HTMLElement>('[data-torque-pane]'))
      pane.hidden = pane.dataset.torquePane !== mode
    for (const option of panel.querySelectorAll('.tri-torque-mode'))
      option.setAttribute('aria-pressed', String(option.getAttribute('data-torque-mode') === mode))
    if (document.activeElement === button)
      panel
        .querySelector<HTMLButtonElement>(
          `[data-torque-pane="${mode}"] .tri-torque-mode[data-torque-mode="${mode}"]`,
        )
        ?.focus({ preventScroll: true })
    const graph = panel.querySelector<SVGSVGElement>('.tri-torque-density')
    if (mode === 'cadence' && graph) showCell(graph, Number(graph.getAttribute('aria-valuenow')))
  }
  const pointer = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const cell = event.target.closest<SVGRectElement>('[data-torque-cell]')
    const graph = cell?.closest<SVGSVGElement>('.tri-torque-density')
    if (!cell || !graph) return
    const cells = Array.from(graph.querySelectorAll('[data-torque-cell]'))
    showCell(graph, cells.indexOf(cell))
  }
  const key = (event: KeyboardEvent): void => {
    if (!(event.target instanceof SVGSVGElement) || !event.target.matches('.tri-torque-density'))
      return
    const graph = event.target
    const index = Number(graph.getAttribute('aria-valuenow'))
    let next: number
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = index + 1
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = index - 1
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = Number(graph.getAttribute('aria-valuemax'))
    else return
    event.preventDefault()
    showCell(graph, next)
  }
  scope.addEventListener('click', click)
  scope.addEventListener('pointermove', pointer)
  scope.addEventListener('pointerdown', pointer)
  scope.addEventListener('keydown', key)
  return () => {
    scope.removeEventListener('click', click)
    scope.removeEventListener('pointermove', pointer)
    scope.removeEventListener('pointerdown', pointer)
    scope.removeEventListener('keydown', key)
  }
}
