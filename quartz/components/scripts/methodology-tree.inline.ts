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

const setupMethodologySteps = () => {
  const steps = document.querySelectorAll<HTMLElement>(STEP_SELECTOR)
  for (const step of steps) {
    const toggle = step.querySelector<HTMLButtonElement>(TOGGLE_SELECTOR)
    const body = step.querySelector<HTMLElement>(BODY_SELECTOR)
    if (!toggle || !body) continue

    const initialOpen = step.dataset.initialOpen !== 'false' && step.classList.contains('is-open')
    setStepState(step, toggle, body, initialOpen)

    const handleToggle = () => {
      const willOpen = !step.classList.contains('is-open')
      setStepState(step, toggle, body, willOpen)
    }

    toggle.addEventListener('click', handleToggle)
    window.addCleanup?.(() => toggle.removeEventListener('click', handleToggle))
  }
}

document.addEventListener('nav', setupMethodologySteps)
