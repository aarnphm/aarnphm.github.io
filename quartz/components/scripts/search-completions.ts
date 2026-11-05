import {
  searchCompletions,
  type SearchCompletion,
  type SearchCompletionOptions,
} from '../../util/search-completion'

export interface SearchCompletions {
  update: () => boolean
  clear: () => void
  dispose: () => void
}

export function createSearchCompletions(
  input: HTMLInputElement,
  host: HTMLElement,
  enabled: () => boolean = () => true,
  options: SearchCompletionOptions & {
    source?: (value: string, cursor: number) => SearchCompletion[]
  } = {},
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
  const anchor = input.closest('.input-container') ?? input
  anchor.after(panel)
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

  // The site's :has() rules turn any attribute flip or insertion here into a restyle of the whole
  // page, so every write is skipped when it would not change anything, and typing that keeps the
  // same suggestions touches no DOM.
  const attr = (element: Element, name: string, value: string | null) => {
    if (element.getAttribute(name) === value) return
    if (value === null) element.removeAttribute(name)
    else element.setAttribute(name, value)
  }
  const same = (a: SearchCompletion[], b: SearchCompletion[]) =>
    a.length === b.length &&
    a.every(
      (choice, i) =>
        choice.insert === b[i].insert &&
        choice.value === b[i].value &&
        choice.operator === b[i].operator &&
        choice.description === b[i].description &&
        choice.from === b[i].from &&
        choice.to === b[i].to,
    )
  const clear = () => {
    choices = []
    if (!panel.hidden) panel.hidden = true
    attr(input, 'aria-expanded', 'false')
    attr(input, 'aria-activedescendant', null)
  }
  // Rows are kept and rewritten in place: a new button, or a textContent write, inserts nodes,
  // while a Text node's data and an inline display are invisible to selectors. Unused rows hide.
  interface Row {
    item: HTMLButtonElement
    token: HTMLElement
    value: HTMLElement
    description: HTMLElement
  }
  const rows: Row[] = []
  const span = (className: string) => {
    const element = document.createElement('span')
    element.className = className
    element.append(document.createTextNode(''))
    return element
  }
  const setText = (element: HTMLElement, text: string) => {
    const node = element.firstChild as Text
    if (node.data !== text) node.data = text
  }
  const option = (index: number): Row => {
    const item = document.createElement('button')
    item.type = 'button'
    item.tabIndex = -1
    item.id = `${list.id}-${index}`
    item.className = 'search-completion'
    item.setAttribute('role', 'option')
    item.setAttribute('aria-selected', 'false')
    const row = {
      item,
      token: span('search-completion-token'),
      value: span('search-completion-value'),
      description: span('search-completion-description'),
    }
    item.append(row.token, row.value, row.description)
    item.addEventListener('pointerdown', event => event.preventDefault())
    item.addEventListener('click', () => apply(index))
    list.append(item)
    return row
  }
  const select = (index: number, reveal = true) => {
    selected = (index + choices.length) % choices.length
    const items = rows.slice(0, choices.length).map(row => row.item)
    items.forEach((item, position) => attr(item, 'aria-selected', String(position === selected)))
    const item = items[selected]
    if (item) {
      attr(input, 'aria-activedescendant', item.id)
      // a forced layout of the whole page; a fresh list already shows its first row
      if (reveal) item.scrollIntoView({ block: 'nearest' })
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
    if (!enabled() || input.selectionStart !== input.selectionEnd) {
      clear()
      return false
    }
    const next = (options.source ?? searchCompletions)(
      input.value,
      input.selectionStart ?? input.value.length,
      options,
    )
    if (!next.length) {
      clear()
      return false
    }
    // the same suggestions as before: keep the list, and the pick, as they are
    if (!panel.hidden && same(next, choices)) return true
    choices = next
    // back to the top before the rebuild: scrollTop after it would force a layout of the page
    if (list.scrollTop) list.scrollTop = 0
    choices.forEach((choice, index) => {
      const row = (rows[index] ??= option(index))
      if (row.item.style.display) row.item.style.display = ''
      if (row.item.title !== choice.insert) row.item.title = choice.insert
      setText(row.token, choice.operator)
      setText(row.value, choice.value)
      setText(row.description, choice.description)
      row.description.style.display = choice.description ? '' : 'none'
    })
    for (const row of rows.slice(choices.length)) row.item.style.display = 'none'
    if (panel.hidden) panel.hidden = false
    attr(input, 'aria-expanded', 'true')
    select(0, false)
    return true
  }
  input.addEventListener(
    'keydown',
    event => {
      if (!choices.length || event.isComposing || event.metaKey || event.altKey) return
      const next = event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === 'n'
      const previous = event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === 'p'
      if (event.ctrlKey && !next && !previous) return
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || next || previous) {
        event.preventDefault()
        event.stopPropagation()
        select(selected + (event.key === 'ArrowDown' || next ? 1 : -1))
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
