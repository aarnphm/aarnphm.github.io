import { searchCompletions, type SearchCompletion } from '../../util/search-completion'

export interface SearchCompletions {
  update: () => boolean
  clear: () => void
  dispose: () => void
}

export function createSearchCompletions(
  input: HTMLInputElement,
  host: HTMLElement,
  enabled: () => boolean = () => true,
): SearchCompletions {
  const abort = new AbortController()
  const { signal } = abort
  const panel = document.createElement('div')
  panel.className = 'search-completions'
  panel.hidden = true
  const label = document.createElement('div')
  label.className = 'search-completions-label'
  label.textContent = 'suggestions'
  const list = document.createElement('div')
  list.className = 'search-completions-list'
  list.id = `search-completions-${crypto.randomUUID()}`
  list.setAttribute('role', 'listbox')
  list.setAttribute('aria-label', 'search suggestions')
  panel.append(label, list)
  input.closest('.input-container')?.after(panel)
  const attributes = [
    'role',
    'aria-autocomplete',
    'aria-controls',
    'aria-expanded',
    'aria-activedescendant',
  ]
  const previousAttributes = new Map(attributes.map(name => [name, input.getAttribute(name)]))
  input.setAttribute('role', 'combobox')
  input.setAttribute('aria-autocomplete', 'list')
  input.setAttribute('aria-controls', list.id)
  let choices: SearchCompletion[] = []
  let selected = 0

  const clear = () => {
    choices = []
    panel.hidden = true
    list.replaceChildren()
    input.setAttribute('aria-expanded', 'false')
    input.removeAttribute('aria-activedescendant')
  }
  const select = (index: number) => {
    selected = (index + choices.length) % choices.length
    const items = [...list.querySelectorAll<HTMLElement>('[role="option"]')]
    items.forEach((item, position) =>
      item.setAttribute('aria-selected', String(position === selected)),
    )
    const item = items[selected]
    if (item) {
      input.setAttribute('aria-activedescendant', item.id)
      item.scrollIntoView({ block: 'nearest' })
    }
  }
  const apply = (index: number) => {
    const choice = choices[index]
    if (!choice) return
    input.setRangeText(choice.insert, choice.from, choice.to, 'end')
    clear()
    input.focus({ preventScroll: true })
    input.dispatchEvent(new Event('input', { bubbles: true }))
  }
  const update = () => {
    clear()
    if (!enabled()) return false
    if (input.selectionStart !== input.selectionEnd) return false
    choices = searchCompletions(input.value, input.selectionStart ?? input.value.length)
    if (!choices.length) return false
    list.replaceChildren(
      ...choices.map((choice, index) => {
        const item = document.createElement('button')
        item.type = 'button'
        item.tabIndex = -1
        item.id = `${list.id}-${index}`
        item.className = 'search-completion'
        item.setAttribute('role', 'option')
        const token = document.createElement('span')
        token.className = 'search-completion-token'
        token.textContent = choice.operator
        const value = document.createElement('span')
        value.className = 'search-completion-value'
        value.textContent = choice.value
        item.append(token, value)
        if (choice.description) {
          const description = document.createElement('span')
          description.className = 'search-completion-description'
          description.textContent = choice.description
          item.append(description)
        }
        item.addEventListener('pointerdown', event => event.preventDefault())
        item.addEventListener('click', () => apply(index))
        return item
      }),
    )
    panel.hidden = false
    input.setAttribute('aria-expanded', 'true')
    select(0)
    return true
  }
  input.addEventListener(
    'keydown',
    event => {
      if (!choices.length || event.isComposing || event.ctrlKey || event.metaKey || event.altKey)
        return
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        event.stopPropagation()
        select(selected + (event.key === 'ArrowDown' ? 1 : -1))
      } else if ((event.key === 'Enter' || event.key === 'Tab') && !event.shiftKey) {
        event.preventDefault()
        event.stopPropagation()
        apply(selected)
      } else if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        clear()
      }
    },
    { signal },
  )
  input.addEventListener('click', update, { signal })
  input.addEventListener(
    'keyup',
    event => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) update()
    },
    { signal },
  )
  input.addEventListener('blur', clear, { signal })
  input.addEventListener('compositionstart', clear, { signal })
  host.addEventListener(
    'toggle',
    () => {
      if (!host.matches(':popover-open')) clear()
    },
    { signal },
  )
  clear()
  return {
    update,
    clear,
    dispose: () => {
      abort.abort()
      panel.remove()
      for (const [name, value] of previousAttributes) {
        if (value === null) input.removeAttribute(name)
        else input.setAttribute(name, value)
      }
    },
  }
}
