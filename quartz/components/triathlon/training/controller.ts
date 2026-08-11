import type { TrainingPlan } from '../../../plugins/stores/training'
import type { TriathlonContext } from '../runtime/context'
import { start } from '../../../functional'
import { clampN } from '../analytics/shared'
import { el } from '../runtime/dom'
import { filterTrainingPlans, initialTrainingModel, updateTraining } from './model'
import { deriveTrainingDocument, type TrainingTreeNode } from './tree'

export const setupTraining = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-training-btn')
  const panel = root.querySelector<HTMLElement>('.tri-training')
  const scrim = root.querySelector<HTMLElement>('.tri-training-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-training-close')
  const title = root.querySelector<HTMLElement>('.tri-training-title')
  const search = root.querySelector<HTMLInputElement>('.tri-training-search')
  const results = root.querySelector<HTMLElement>('.tri-training-results')
  const list = root.querySelector<HTMLElement>('.tri-training-plans')
  const tree = root.querySelector<HTMLElement>('.tri-training-tree')
  const preview = root.querySelector<HTMLElement>('.tri-training-doc')
  const pageMode = root.dataset.triView === 'training'
  if (!panel || (!btn && !pageMode)) return null

  let selIndex = -1

  const showPlan = (plan: TrainingPlan) => {
    if (!preview) return
    const head = el('div', 'tri-pop-head tri-training-head')
    head.appendChild(el('span', 'tri-pop-date tri-training-meta-name', plan.meta))
    const meta = el('ul', 'tri-training-meta')
    const metaRow = (label: string, value: string) => {
      if (!value) return
      const li = el('li')
      li.append(el('span', 'tri-training-meta-k', label), el('span', 'tri-training-meta-v', value))
      meta.appendChild(li)
    }
    metaRow('distance', plan.distance)
    metaRow('date', plan.date)
    metaRow('objectif', plan.target)
    metaRow(
      'avec',
      plan.author
        ? plan.author
            .split(',')
            .map(s => s.trim())
            .join(', ')
        : '',
    )
    if (meta.childElementCount) head.appendChild(meta)
    const body = el('div', 'tri-training-render')
    const renderDocument = deriveTrainingDocument(plan)
    body.innerHTML = renderDocument.html
    preview.replaceChildren(head, body)
    preview.scrollTo({ top: 0 })
    buildTree(renderDocument.tree)
    document.dispatchEvent(
      new CustomEvent('contentdecrypted', { detail: { article: preview, content: preview } }),
    )
  }

  const buildTree = (roots: readonly TrainingTreeNode[]) => {
    if (!tree) return
    tree.replaceChildren()
    if (!roots.length) return
    const nb = ' '
    const seg = (bar: boolean) => (bar ? `│${nb.repeat(3)}` : nb.repeat(4))
    const lines: HTMLElement[] = []
    const walk = (node: TrainingTreeNode, last: boolean, anc: boolean[]) => {
      const prefix = anc.map(seg).join('') + (last ? '└── ' : '├── ')
      const line = el('div', 'tri-training-tree-line')
      line.appendChild(el('span', 'tri-training-tree-prefix', prefix))
      const link = el('button', 'tri-training-tree-link', node.label)
      link.setAttribute('type', 'button')
      link.dataset.target = node.id
      line.appendChild(link)
      lines.push(line)
      node.children.forEach((c, i) => walk(c, i === node.children.length - 1, [...anc, !last]))
    }
    roots.forEach((n, i) => walk(n, i === roots.length - 1, []))
    tree.replaceChildren(...lines)
  }

  const select = (idx: number) => {
    const plans = program.retrieve().plans
    if (!plans.length) return
    selIndex = clampN(idx, 0, plans.length - 1)
    program.dispatch({ type: 'select-plan', id: plans[selIndex].id })
  }

  const ritem = (plan: TrainingPlan, i: number): HTMLElement => {
    const it = el('button', 'tri-ana-ritem')
    it.setAttribute('type', 'button')
    it.dataset.plan = String(i)
    it.append(
      el('span', 'tri-ana-ritem-t', plan.meta),
      el('span', 'tri-ana-ritem-s', [plan.distance, plan.target].filter(Boolean).join(' · ')),
    )
    return it
  }

  const renderList = () => {
    if (!list) return
    const plans = program.retrieve().plans
    list.replaceChildren(...plans.map(ritem))
    if (plans.length) select(0)
    else preview?.replaceChildren(el('div', 'tri-ana-empty', context.formatter.text('no plan')))
  }

  const renderSelection = (id: string): void => {
    const plans = program.retrieve().plans
    const index = plans.findIndex(plan => plan.id === id)
    if (index < 0) return
    selIndex = index
    const items = list ? Array.from(list.querySelectorAll<HTMLElement>('.tri-ana-ritem')) : []
    items.forEach((item, itemIndex) =>
      item.classList.toggle('tri-ana-ritem--sel', itemIndex === index),
    )
    showPlan(plans[index])
  }

  const program = start({
    init: () => ({ model: initialTrainingModel(), effects: [] }),
    reduce: updateTraining,
    effects: (effect, state) => {
      if (effect.type === 'load-artifact') {
        const path = root.dataset.trainingPath
        if (!path) {
          state.dispatch({ type: 'failed' })
          return
        }
        void context.resources.training.load(path).then(result => {
          if (result.status === 'ready')
            state.dispatch({ type: 'loaded', plans: result.value.plans })
          else if (result.status === 'error') state.dispatch({ type: 'failed' })
        })
      } else if (effect.type === 'render-plans') {
        if (pageMode && list?.dataset.triSsr === 'true' && preview?.dataset.triSsr === 'true') {
          selIndex = 0
        } else renderList()
      } else if (effect.type === 'render-plan') {
        renderSelection(effect.id)
      } else if (effect.type === 'render-search') {
        renderSearch()
      } else {
        scrollToTarget(effect.id)
      }
    },
  })

  const load = (): void => {
    if (program.retrieve().status === 'idle') program.dispatch({ type: 'load' })
  }

  const toMain = () => {
    if (search) search.value = ''
    program.dispatch({ type: 'query', value: '' })
  }
  const open = () => {
    toMain()
    root.classList.add('tri-training-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
    panel.focus({ preventScroll: true })
  }
  const close = () => {
    const wasOpen = root.classList.contains('tri-training-open')
    root.classList.remove('tri-training-open')
    panel.setAttribute('aria-hidden', 'true')
    if (wasOpen && !pageMode) btn?.focus({ preventScroll: true })
  }

  const renderSearch = () => {
    if (!search || !results) return
    const model = program.retrieve()
    const q = model.query.trim().toLowerCase()
    if (!q) {
      panel.classList.remove('tri-training--searching')
      results.replaceChildren()
      results.setAttribute('aria-hidden', 'true')
      return
    }
    panel.classList.add('tri-training--searching')
    results.setAttribute('aria-hidden', 'false')
    const hitIds = new Set(filterTrainingPlans(model.plans, q).map(plan => plan.id))
    const hits = model.plans.map((p, i) => ({ p, i })).filter(({ p }) => hitIds.has(p.id))
    results.replaceChildren(
      ...(hits.length
        ? hits.map(({ p, i }) => ritem(p, i))
        : [el('div', 'tri-ana-empty', context.formatter.text('no matches'))]),
    )
  }
  const runSearch = (): void => program.dispatch({ type: 'query', value: search?.value ?? '' })

  const activate = (it?: HTMLElement | null) => {
    if (!it || it.dataset.plan == null) return
    select(Number(it.dataset.plan))
    toMain()
  }
  const eventItem = (event: MouseEvent, selector: string): HTMLElement | null =>
    event.target instanceof Element ? event.target.closest(selector) : null
  const onListClick = (event: MouseEvent) => activate(eventItem(event, '.tri-ana-ritem'))
  const onResultsClick = (event: MouseEvent) => activate(eventItem(event, '.tri-ana-ritem'))
  const scrollToTarget = (id: string): void => {
    if (!preview) return
    const target = preview.querySelector<HTMLElement>(`[id="${CSS.escape(id)}"]`)
    if (!target) return
    preview.scrollTo({
      top:
        preview.scrollTop +
        target.getBoundingClientRect().top -
        preview.getBoundingClientRect().top -
        8,
      behavior: 'smooth',
    })
  }
  const onTreeClick = (event: MouseEvent) => {
    const target = eventItem(event, '[data-target]')?.dataset.target
    if (target) program.dispatch({ type: 'select-tree-target', id: target })
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape' || (!pageMode && !root.classList.contains('tri-training-open')))
      return
    if (search && search.value) {
      search.value = ''
      runSearch()
      return
    }
    if (pageMode) return
    close()
  }

  if (pageMode) {
    load()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
  }
  search?.addEventListener('input', runSearch)
  results?.addEventListener('click', onResultsClick)
  list?.addEventListener('click', onListClick)
  tree?.addEventListener('click', onTreeClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    results?.removeEventListener('click', onResultsClick)
    list?.removeEventListener('click', onListClick)
    tree?.removeEventListener('click', onTreeClick)
    document.removeEventListener('keydown', onKey)
    program.stop()
  }
}
