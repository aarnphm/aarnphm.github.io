import { registerEscapeHandler } from './util'

interface PageListState {
  container: HTMLElement
  input: HTMLInputElement
  status: HTMLOutputElement
  clearBtn: HTMLButtonElement
  items: HTMLLIElement[]
  focusedIdx: number
}

interface TagSectionState {
  container: HTMLElement
  input: HTMLInputElement
  status: HTMLOutputElement
  clearBtn: HTMLButtonElement
  sections: HTMLElement[]
  focusedIdx: number
}

function extractTagsFromQuery(query: string): string[] {
  const matches = query.match(/#[\w/-]+/g) || []
  return matches.map(t => t.slice(1).toLowerCase())
}

function extractTextQuery(query: string): string {
  return query
    .replace(/#[\w/-]+/g, '')
    .trim()
    .toLowerCase()
}

function tagMatches(queryTag: string, itemTags: string[]): boolean {
  return itemTags.some(t => t.includes(queryTag))
}

function debounce<T extends (...args: any[]) => any>(fn: T, delay: number): T {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  return ((...args: Parameters<T>) => {
    if (timeoutId) clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }) as T
}

function getVisibleItems(state: PageListState): HTMLLIElement[] {
  return state.items.filter(li => !li.hidden)
}

function updateItemFocus(state: PageListState) {
  state.items.forEach(li => li.classList.remove('focus'))
  const visible = getVisibleItems(state)
  if (visible.length === 0) {
    state.focusedIdx = -1
    return
  }
  state.focusedIdx = Math.max(0, Math.min(state.focusedIdx, visible.length - 1))
  const focused = visible[state.focusedIdx]
  if (focused) {
    focused.classList.add('focus')
    focused.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }
}

function navigateItemNext(state: PageListState) {
  const visible = getVisibleItems(state)
  if (visible.length === 0) return
  state.focusedIdx = (state.focusedIdx + 1) % visible.length
  updateItemFocus(state)
}

function navigateItemPrev(state: PageListState) {
  const visible = getVisibleItems(state)
  if (visible.length === 0) return
  state.focusedIdx = (state.focusedIdx - 1 + visible.length) % visible.length
  updateItemFocus(state)
}

function openFocusedItem(state: PageListState) {
  const visible = getVisibleItems(state)
  const focused = visible[state.focusedIdx]
  const anchor = focused?.querySelector('a.note-link') as HTMLAnchorElement | null
  if (anchor) anchor.click()
}

function updateItemStatus(state: PageListState, visible: number, total: number, query: string) {
  state.status.textContent = query ? `${visible} of ${total}` : ''
}

function updateClearButton(state: PageListState | TagSectionState, query: string) {
  state.clearBtn.style.display = query ? 'flex' : 'none'
}

function filterItems(state: PageListState, query: string) {
  const { items } = state
  const total = items.length

  if (!query.trim()) {
    items.forEach(li => (li.hidden = false))
    updateItemStatus(state, total, total, '')
    updateClearButton(state, '')
    state.focusedIdx = -1
    items.forEach(li => li.classList.remove('focus'))
    return
  }

  const queryTags = extractTagsFromQuery(query)
  const textQuery = extractTextQuery(query)

  let visible = 0
  items.forEach(li => {
    const title = (li.dataset.title ?? '').toLowerCase()
    const itemTags: string[] = JSON.parse(li.dataset.tags ?? '[]').map((t: string) =>
      t.toLowerCase(),
    )

    const textMatch = !textQuery || title.includes(textQuery)
    const tagMatch = queryTags.some(qt => tagMatches(qt, itemTags))

    li.hidden = !(textMatch && tagMatch)
    if (!li.hidden) visible++
  })

  state.focusedIdx = visible > 0 ? 0 : -1
  updateItemFocus(state)
  updateItemStatus(state, visible, total, query)
  updateClearButton(state, query)
}

function getVisibleSections(state: TagSectionState): HTMLElement[] {
  return state.sections.filter(s => !s.hidden)
}

function updateSectionFocus(state: TagSectionState) {
  state.sections.forEach(s => s.classList.remove('focus'))
  const visible = getVisibleSections(state)
  if (visible.length === 0) {
    state.focusedIdx = -1
    return
  }
  state.focusedIdx = Math.max(0, Math.min(state.focusedIdx, visible.length - 1))
  const focused = visible[state.focusedIdx]
  if (focused) {
    focused.classList.add('focus')
    focused.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }
}

function navigateSectionNext(state: TagSectionState) {
  const visible = getVisibleSections(state)
  if (visible.length === 0) return
  state.focusedIdx = (state.focusedIdx + 1) % visible.length
  updateSectionFocus(state)
}

function navigateSectionPrev(state: TagSectionState) {
  const visible = getVisibleSections(state)
  if (visible.length === 0) return
  state.focusedIdx = (state.focusedIdx - 1 + visible.length) % visible.length
  updateSectionFocus(state)
}

function openFocusedSection(state: TagSectionState) {
  const visible = getVisibleSections(state)
  const focused = visible[state.focusedIdx]
  const anchor = focused?.querySelector('a.tag-link') as HTMLAnchorElement | null
  if (anchor) anchor.click()
}

function updateSectionStatus(
  state: TagSectionState,
  visible: number,
  total: number,
  query: string,
) {
  state.status.textContent = query ? `${visible} of ${total} tags` : ''
}

function filterSections(state: TagSectionState, query: string) {
  const { sections } = state
  const total = sections.length

  if (!query.trim()) {
    sections.forEach(s => (s.hidden = false))
    updateSectionStatus(state, total, total, '')
    updateClearButton(state, '')
    state.focusedIdx = -1
    sections.forEach(s => s.classList.remove('focus'))
    return
  }

  const textQuery = query.trim().toLowerCase()

  let visible = 0
  sections.forEach(section => {
    const tagLink = section.querySelector('a.tag-link')
    const tagName = (tagLink?.textContent ?? '').toLowerCase()
    const match = tagName.includes(textQuery)
    section.hidden = !match
    if (match) visible++
  })

  state.focusedIdx = visible > 0 ? 0 : -1
  updateSectionFocus(state)
  updateSectionStatus(state, visible, total, query)
  updateClearButton(state, query)
}

function setupAllTagsSearch(container: HTMLElement) {
  const input = container.querySelector<HTMLInputElement>('.page-list-search-input')
  const status = container.querySelector<HTMLOutputElement>('.page-list-search-status')
  const clearBtn = container.querySelector<HTMLButtonElement>('.page-list-search-clear')

  if (!input || !status || !clearBtn) return

  const parent = container.parentElement
  if (!parent) return

  const sectionsContainer = parent.querySelector(':scope > div:last-child')
  if (!sectionsContainer) return

  const sections = Array.from(sectionsContainer.querySelectorAll<HTMLElement>(':scope > div'))
  if (sections.length === 0) return

  const state: TagSectionState = { container, input, status, clearBtn, sections, focusedIdx: -1 }

  const debouncedFilter = debounce((query: string) => filterSections(state, query), 50)

  input.addEventListener('input', () => debouncedFilter(input.value))

  input.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      input.value = ''
      input.blur()
      filterSections(state, '')
      return
    }
    if (e.ctrlKey && e.key === 'n') {
      e.preventDefault()
      navigateSectionNext(state)
      return
    }
    if (e.ctrlKey && e.key === 'p') {
      e.preventDefault()
      navigateSectionPrev(state)
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      navigateSectionNext(state)
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      navigateSectionPrev(state)
      return
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      openFocusedSection(state)
      return
    }
  })

  clearBtn.addEventListener('click', () => {
    input.value = ''
    input.focus()
    filterSections(state, '')
  })

  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k' && !e.shiftKey) {
      const active = document.activeElement
      const isOtherInput =
        (active?.tagName === 'INPUT' || active?.tagName === 'TEXTAREA') && active !== input
      if (isOtherInput) return
      e.preventDefault()
      input.focus()
      input.select()
    }
  })

  registerEscapeHandler(container, () => {
    input.value = ''
    filterSections(state, '')
  })
}

function setupPageListSearch(container: HTMLElement) {
  const input = container.querySelector<HTMLInputElement>('.page-list-search-input')
  const status = container.querySelector<HTMLOutputElement>('.page-list-search-status')
  const clearBtn = container.querySelector<HTMLButtonElement>('.page-list-search-clear')

  if (!input || !status || !clearBtn) return

  const listContainer = container.closest('[data-pagelist]') ?? container.parentElement
  if (!listContainer) return

  const items = Array.from(listContainer.querySelectorAll<HTMLLIElement>('.section-li'))
  if (items.length === 0) return

  const state: PageListState = { container, input, status, clearBtn, items, focusedIdx: -1 }

  const debouncedFilter = debounce((query: string) => filterItems(state, query), 50)

  input.addEventListener('input', () => debouncedFilter(input.value))

  input.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      input.value = ''
      input.blur()
      filterItems(state, '')
      return
    }
    if (e.ctrlKey && e.key === 'n') {
      e.preventDefault()
      navigateItemNext(state)
      return
    }
    if (e.ctrlKey && e.key === 'p') {
      e.preventDefault()
      navigateItemPrev(state)
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      navigateItemNext(state)
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      navigateItemPrev(state)
      return
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      openFocusedItem(state)
      return
    }
  })

  clearBtn.addEventListener('click', () => {
    input.value = ''
    input.focus()
    filterItems(state, '')
  })

  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k' && !e.shiftKey) {
      const active = document.activeElement
      const isOtherInput =
        (active?.tagName === 'INPUT' || active?.tagName === 'TEXTAREA') && active !== input
      if (isOtherInput) return
      e.preventDefault()
      input.focus()
      input.select()
    }
  })

  registerEscapeHandler(container, () => {
    input.value = ''
    filterItems(state, '')
  })
}

document.addEventListener('nav', () => {
  const containers = document.querySelectorAll<HTMLElement>('.page-list-search-container')
  containers.forEach(container => {
    if (container.dataset.allTag === 'true') {
      setupAllTagsSearch(container)
    } else {
      setupPageListSearch(container)
    }
  })
})
