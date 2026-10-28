import { mooncakePlan, mooncakeStage, mooncakeView } from '../../util/mooncake-flow'
import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'

const render = (root: HTMLElement) => {
  const plan = mooncakePlan(root.dataset.mckRequest)
  const stage = mooncakeStage(root.dataset.mckStage)
  const view = mooncakeView(plan, stage)
  for (const cell of root.querySelectorAll<SVGRectElement>('[data-mck-cell]'))
    cell.dataset.state = view.cells[cell.dataset.mckCell!] ?? 'empty'
  for (const el of root.querySelectorAll<SVGElement>('[data-mck-active-key]'))
    el.dataset.active = view.active[el.dataset.mckActiveKey!] ? 'true' : 'false'
  const values: Record<string, string> = {}
  for (const route of plan.prefill) {
    values[`${route.id}-hit`] = `${route.hit}/${plan.request.blocks.length}`
    values[`${route.id}-ttft`] = `${route.ttftMs} ms`
  }
  for (const route of plan.decode) values[`${route.id}-after`] = String(route.after)
  for (const el of root.querySelectorAll<HTMLElement>('[data-mck-val]')) {
    const text = values[el.dataset.mckVal!]
    if (text !== undefined && el.textContent !== text) el.textContent = text
  }
  for (const row of root.querySelectorAll<HTMLElement>('[data-mck-row]')) {
    const id = row.dataset.mckRow
    row.dataset.selected = id === plan.prefillId || id === plan.decodeId ? 'true' : 'false'
  }
}

registerMdxContent('[data-mooncake-flow]', root => {
  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement
    if (input.matches('[data-mck-request-input]')) root.dataset.mckRequest = input.value
    else if (input.matches('[data-mck-stage-input]')) root.dataset.mckStage = input.value
    else return
    render(root)
  }
  root.addEventListener('change', onChange)
  window.addCleanup(() => root.removeEventListener('change', onChange))
  window.addCleanup(observeSvgScale(root.querySelectorAll<SVGSVGElement>('.mck-graph'), '--mck-u'))
  render(root)
})
