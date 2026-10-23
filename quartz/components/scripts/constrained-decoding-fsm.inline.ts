import { registerMdxContent } from './mdx-content'

type CdfsmView = {
  state: number
  visited: number[]
  seen: string[]
  taken: string[]
  status: string
}

const cdfsmParse = (root: HTMLElement): CdfsmView[] => {
  try {
    const parsed = JSON.parse(root.dataset.cdfsmViews ?? '[]')
    return Array.isArray(parsed) ? (parsed as CdfsmView[]) : []
  } catch {
    return []
  }
}

const cdfsmRender = (root: HTMLElement, views: CdfsmView[], step: number) => {
  const view = views[step]
  const last = views.length - 1
  root.dataset.cdfsmStep = String(step)

  for (const el of root.querySelectorAll<SVGGElement>('[data-cdfsm-state]')) {
    const id = Number(el.dataset.cdfsmState)
    el.classList.toggle('is-current', id === view.state)
    el.classList.toggle('is-visited', view.visited.includes(id))
  }
  for (const el of root.querySelectorAll<SVGGElement>('[data-cdfsm-edge]')) {
    const id = el.dataset.cdfsmEdge ?? ''
    el.classList.toggle('is-taken', view.taken.includes(id))
    el.classList.toggle('is-seen', view.seen.includes(id))
  }
  for (const el of root.querySelectorAll<HTMLElement>('[data-cdfsm-tok]')) {
    const i = Number(el.dataset.cdfsmTok)
    el.hidden = i >= step
    el.classList.toggle('is-new', i === step - 1)
  }
  for (const el of root.querySelectorAll<HTMLElement>('[data-cdfsm-panel]')) {
    el.hidden = Number(el.dataset.cdfsmPanel) !== step
  }
  for (const btn of root.querySelectorAll<HTMLButtonElement>('[data-cdfsm-action]')) {
    const action = btn.dataset.cdfsmAction
    const off = (action === 'back' && step === 0) || (action === 'step' && step === last)
    btn.setAttribute('aria-disabled', String(off))
  }
  const status = root.querySelector<HTMLElement>('[data-cdfsm-status]')
  if (status) status.textContent = view.status
}

registerMdxContent('[data-cdfsm]', root => {
  const views = cdfsmParse(root)
  if (!views.length) return
  const last = views.length - 1
  let step = 0

  const go = (next: number) => {
    const clamped = Math.max(0, Math.min(last, next))
    if (clamped === step) return
    step = clamped
    cdfsmRender(root, views, step)
  }

  // aria-disabled keeps focus on the button at either end; the click is ignored instead.
  const onClick = (event: MouseEvent) => {
    const btn = (event.target as Element | null)?.closest<HTMLButtonElement>('[data-cdfsm-action]')
    if (!btn || !root.contains(btn)) return
    const action = btn.dataset.cdfsmAction
    if (action === 'step') go(step + 1)
    else if (action === 'back') go(step - 1)
    else if (action === 'reset') go(0)
  }

  const onKey = (event: KeyboardEvent) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return
    const moves: Record<string, number> = {
      ArrowRight: step + 1,
      ArrowLeft: step - 1,
      Home: 0,
      End: last,
    }
    if (!(event.key in moves)) return
    event.preventDefault()
    go(moves[event.key])
  }

  const controls = root.querySelector<HTMLElement>('.cdfsm-controls')
  root.addEventListener('click', onClick)
  controls?.addEventListener('keydown', onKey)
  cdfsmRender(root, views, step)

  window.addCleanup(() => {
    root.removeEventListener('click', onClick)
    controls?.removeEventListener('keydown', onKey)
  })
})
