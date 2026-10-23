import { registerMdxContent } from './mdx-content'

const STEP_SELECTOR = '.methodology-step'
const TOGGLE_SELECTOR = '[data-step-toggle]'
const BODY_SELECTOR = '[data-step-body]'

const setStepState = (
  step: HTMLElement,
  toggle: HTMLButtonElement,
  body: HTMLElement,
  open: boolean,
) => {
  step.classList.toggle('is-open', open)
  toggle.setAttribute('aria-expanded', open ? 'true' : 'false')
  body.hidden = !open
}

registerMdxContent(STEP_SELECTOR, step => {
  const toggle = step.querySelector<HTMLButtonElement>(TOGGLE_SELECTOR)
  const body = step.querySelector<HTMLElement>(BODY_SELECTOR)
  if (!toggle || !body) return

  const initialOpen = step.dataset.initialOpen !== 'false' && step.classList.contains('is-open')
  setStepState(step, toggle, body, initialOpen)

  const handleToggle = () => {
    const willOpen = !step.classList.contains('is-open')
    setStepState(step, toggle, body, willOpen)
  }

  toggle.addEventListener('click', handleToggle)
  window.addCleanup?.(() => toggle.removeEventListener('click', handleToggle))
})
