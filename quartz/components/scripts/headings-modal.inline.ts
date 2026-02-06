import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import { registerEscapeHandler } from './util'

let ag: RoughAnnotation | null = null

interface HeadingInfo {
  element: HTMLDivElement
  level: number
  text: string
  uniqueText: string
  line: number
}

let modal: HTMLElement | null = null
let isOpen = false
let allHeadings: HeadingInfo[] = []
let filteredHeadings: HeadingInfo[] = []
let currentIndex = 0
let isSecondaryMode = false
let secondaryGroups: Record<string, number[]> = {}
let secondaryExtmarks: HTMLElement[] = []
let secondaryKeys: string[] = []
let searchInput: HTMLInputElement | null = null
let searchQuery = ''

const SECONDARY_CHOICE_KEYS = [
  'a',
  's',
  'd',
  'f',
  'l',
  ';',
  'h',
  'g',
  'u',
  'i',
  'o',
  'p',
  'w',
  'e',
  'r',
  't',
  'y',
  'c',
  'v',
  'b',
  'n',
  'm',
  'x',
  'z',
]

function shouldIgnoreShortcutTarget(target: EventTarget | null): boolean {
  let el: Element | null = target instanceof Element ? target : null
  if (!el) {
    const active = document.activeElement
    el = active instanceof Element ? active : null
  }

  if (!el) return false

  const tag = el.tagName.toLowerCase()
  if (tag === 'input' || tag === 'textarea') return true
  if ((el as HTMLElement).isContentEditable) return true
  if (el.closest('.search .search-container')) return true
  if (el.closest('.stream-search-container')) return true

  return false
}

function getVisibleHeadings(): HeadingInfo[] {
  return filteredHeadings.length > 0 || searchQuery.trim().length > 0
    ? filteredHeadings
    : allHeadings
}

function updateFilteredHeadings() {
  const query = searchQuery.trim().toLowerCase()
  if (query.length === 0) {
    filteredHeadings = [...allHeadings]
  } else {
    filteredHeadings = allHeadings.filter(heading => heading.text.toLowerCase().includes(query))
  }

  const visible = getVisibleHeadings()
  secondaryGroups = buildLetterGroups(visible)
  if (visible.length === 0) {
    currentIndex = 0
  } else if (currentIndex >= visible.length) {
    currentIndex = visible.length - 1
  }

  renderHeadings()
}

function extractHeadings(): HeadingInfo[] {
  const headingSelectors =
    '.page-content h2, .page-content h3, .page-content h4, .page-content h5, .page-content h6'
  const elements = Array.from(document.querySelectorAll(headingSelectors))
  const textCounts = new Map<string, number>()

  return elements
    .map((el, index) => {
      const level = parseInt(el.tagName.charAt(1))
      const text = el.textContent?.trim() || ''

      if (text.length === 0) {
        return null
      }

      const count = textCounts.get(text) || 0
      textCounts.set(text, count + 1)
      const uniqueText = count > 0 ? `${text} (${count + 1})` : text

      return { element: el as HTMLElement, level, text, uniqueText, line: index }
    })
    .filter((h): h is HeadingInfo => h !== null)
}

function buildLetterGroups(headings: HeadingInfo[]): Record<string, number[]> {
  const groups: Record<string, number[]> = {}

  headings.forEach((heading, index) => {
    const initials = new Set<string>()

    // Function to add initial letters
    function addInitial(str: string) {
      if (!str) return
      const firstLetter = str.match(/[a-zA-Z]/)
      if (firstLetter) {
        const letter = firstLetter[0].toLowerCase()
        initials.add(letter)
      }
    }

    const text = heading.text.trim()
    addInitial(text)

    initials.forEach(letter => {
      if (!groups[letter]) groups[letter] = []
      if (!groups[letter].includes(index)) {
        groups[letter].push(index)
      }
    })
  })

  return groups
}

function renderHeadings() {
  const listContainer = modal?.querySelector('.headings-list')
  if (!listContainer) return

  listContainer.innerHTML = ''

  const visible = getVisibleHeadings()

  if (visible.length === 0) {
    const empty = document.createElement('div')
    empty.className = 'heading-item heading-item-empty'
    empty.textContent = 'no headings match'
    listContainer.appendChild(empty)
    return
  }

  visible.forEach((heading, index) => {
    const item = document.createElement('div')
    item.className = 'heading-item'
    item.dataset.index = index.toString()

    const indent = Math.max(heading.level - 1, 0) * 2
    item.style.paddingLeft = `${indent}ch`

    if (index === currentIndex) {
      item.classList.add('active')
    }

    item.textContent = heading.uniqueText
    item.addEventListener('click', () => jumpToHeading(index))

    listContainer.appendChild(item)
  })
}

function updateActiveItem() {
  const items = modal?.querySelectorAll('.heading-item')
  if (!items) return

  items.forEach((item, index) => {
    item.classList.toggle(
      'active',
      index === currentIndex && !item.classList.contains('heading-item-empty'),
    )
  })

  const activeItem = items[currentIndex]
  if (activeItem) {
    activeItem.scrollIntoView({ block: 'nearest' })
  }
}

function jumpToHeading(index: number) {
  const visible = getVisibleHeadings()
  if (index < 0 || index >= visible.length) return

  const heading = visible[index]
  closeModal()

  if (ag) ag.hide()

  const highlight = heading.element.querySelector('span.highlight-span') as HTMLElement
  if (highlight) {
    ag = annotate(highlight, {
      type: 'box',
      color: 'rgba(234, 157, 52, 0.45)',
      animate: false,
      multiline: true,
      brackets: ['left', 'right'],
    })
    setTimeout(() => ag!.show(), 50)
    setTimeout(() => ag?.hide(), 2500)
  }

  heading.element.style.outline = 'none'

  const headingRect = heading.element.getBoundingClientRect()
  const absoluteTop = window.pageYOffset + headingRect.top
  const middle = absoluteTop - window.innerHeight / 2 + headingRect.height / 2

  window.scrollTo({ top: middle, behavior: 'smooth' })
}

function clearSecondaryMode() {
  secondaryExtmarks.forEach(el => el.remove())
  secondaryExtmarks = []
  secondaryKeys = []
  isSecondaryMode = false
}

function findKeyLetterIndex(text: string, key: string): number {
  // Find the first occurrence of the key letter in the text
  const lowerText = text.toLowerCase()
  const lowerKey = key.toLowerCase()

  for (let i = 0; i < text.length; i++) {
    if (lowerText[i] === lowerKey && /[a-zA-Z]/.test(text[i])) {
      return i
    }
  }

  return -1
}

function enterSecondaryMode(letter: string) {
  const indices = secondaryGroups[letter]
  if (!indices || indices.length <= 1) return

  clearSecondaryMode()
  isSecondaryMode = true

  // Sort by distance from current middle
  const listContainer = modal?.querySelector('.headings-list')
  if (!listContainer) return

  const visible = getVisibleHeadings()
  const middle = Math.ceil(visible.length / 2)
  indices.sort((a, b) => {
    const distA = Math.abs(a - middle)
    const distB = Math.abs(b - middle)
    if (distA !== distB) return distA - distB
    return b - a // Prefer later items if distance is same
  })

  // Assign keys
  const assigned: Record<number, string> = {}
  indices.forEach((index, i) => {
    if (i >= SECONDARY_CHOICE_KEYS.length) return
    const key = SECONDARY_CHOICE_KEYS[i]
    assigned[index] = key
    secondaryKeys.push(key)
  })

  Object.entries(assigned).forEach(([indexStr, key]) => {
    const index = parseInt(indexStr)
    const item = listContainer.querySelector(`[data-index="${index}"]`) as HTMLElement
    if (!item) return

    const text = visible[index]?.uniqueText || item.textContent || ''
    const keyIndex = findKeyLetterIndex(text, key)

    if (keyIndex !== -1) {
      // Wrap the key letter with a mark element
      const before = text.substring(0, keyIndex)
      const keyChar = text.substring(keyIndex, keyIndex + 1)
      const after = text.substring(keyIndex + 1)

      item.innerHTML = ''
      if (before) item.appendChild(document.createTextNode(before))

      const mark = document.createElement('mark')
      mark.className = 'key-extmark'
      mark.textContent = keyChar
      item.appendChild(mark)
      secondaryExtmarks.push(mark)

      if (after) item.appendChild(document.createTextNode(after))
    } else {
      // Fallback: prepend the key if not found in text
      const mark = document.createElement('mark')
      mark.className = 'key-extmark'
      mark.textContent = key
      item.prepend(mark)
      item.prepend(document.createTextNode(' '))
      secondaryExtmarks.push(mark)
    }
  })

  const handleSecondaryKey = (e: KeyboardEvent) => {
    if (!isSecondaryMode) return

    const key = e.key.toLowerCase()
    if (secondaryKeys.includes(key)) {
      e.preventDefault()
      e.stopPropagation()

      const index = Object.entries(assigned).find(([, k]) => k === key)?.[0]
      if (index !== undefined) {
        jumpToHeading(parseInt(index))
      }
    } else if (e.key === 'Escape') {
      e.preventDefault()
      clearSecondaryMode()
      renderHeadings() // Re-render to clear marks
    }
  }

  document.addEventListener('keydown', handleSecondaryKey, true)
  window.addCleanup(() => document.removeEventListener('keydown', handleSecondaryKey, true))
}

function handleLetterKey(letter: string) {
  if (isSecondaryMode) return

  const indices = secondaryGroups[letter]
  if (!indices || indices.length === 0) return

  if (indices.length === 1) {
    jumpToHeading(indices[0])
  } else {
    enterSecondaryMode(letter)
  }
}

function openModal() {
  if (isOpen) return

  allHeadings = extractHeadings()
  if (allHeadings.length === 0) return

  searchQuery = ''
  if (searchInput) {
    searchInput.value = ''
  }

  filteredHeadings = [...allHeadings]
  currentIndex = 0
  updateFilteredHeadings()
  isOpen = true

  if (!modal) {
    modal = document.querySelector('.headings-modal-container')
  }

  if (modal) {
    modal.style.display = 'flex'
    const modalContent = modal.querySelector('.headings-modal') as HTMLElement
    modalContent?.focus()
  }
}

function closeModal() {
  if (!isOpen) return

  clearSecondaryMode()
  isOpen = false

  if (modal) {
    modal.style.display = 'none'
  }

  document.body.focus()
}

function handleKeyDown(e: KeyboardEvent) {
  if (!isOpen) return

  if (searchInput && e.target === searchInput) {
    return
  }

  if (isSecondaryMode) {
    // Secondary mode keys are handled by their own listener
    return
  }

  switch (e.key) {
    case 'Escape':
      e.preventDefault()
      closeModal()
      break

    case 'Enter':
      e.preventDefault()
      jumpToHeading(currentIndex)
      break

    case 'ArrowDown':
    case 'j':
      e.preventDefault()
      currentIndex = Math.min(currentIndex + 1, Math.max(getVisibleHeadings().length - 1, 0))
      updateActiveItem()
      break

    case 'ArrowUp':
    case 'k':
      e.preventDefault()
      currentIndex = Math.max(currentIndex - 1, 0)
      updateActiveItem()
      break

    case '/':
      e.preventDefault()
      if (searchInput) {
        searchInput.focus()
        searchInput.select()
      }
      break

    default:
      // Handle letter keys
      if (
        e.key.length === 1 &&
        e.key.match(/[a-z]/i) &&
        !e.ctrlKey &&
        !e.metaKey &&
        e.key !== '/'
      ) {
        const letter = e.key.toLowerCase()
        if (letter !== 'j' && letter !== 'k') {
          // Don't interfere with navigation
          e.preventDefault()
          handleLetterKey(letter)
        }
      }
      break
  }
}

function handleGlobalKeyDown(e: KeyboardEvent) {
  // Don't trigger if modal is already open
  if (isOpen) return
  if (shouldIgnoreShortcutTarget(e.target)) return

  // Check for 'gh' sequence
  if (e.key === 'g' && !e.ctrlKey && !e.metaKey && !e.altKey) {
    // Set a temporary flag to wait for 'h'
    let waitingForH = true
    const timeout = setTimeout(() => {
      waitingForH = false
    }, 1000)

    const handleH = (e2: KeyboardEvent) => {
      if (shouldIgnoreShortcutTarget(e2.target)) {
        waitingForH = false
        clearTimeout(timeout)
        return
      }
      if (waitingForH && e2.key === 'h') {
        e2.preventDefault()
        clearTimeout(timeout)
        document.removeEventListener('keydown', handleH)
        openModal()
      }
      waitingForH = false
    }

    document.addEventListener('keydown', handleH, { once: true })
    window.addCleanup(() => {
      clearTimeout(timeout)
      document.removeEventListener('keydown', handleH)
    })
  }
}

document.addEventListener('nav', () => {
  modal = document.querySelector('.headings-modal-container')

  if (modal) {
    const modalContent = modal.querySelector('.headings-modal') as HTMLElement
    if (modalContent) {
      modalContent.setAttribute('tabindex', '-1')
    }

    const header = modal.querySelector('.headings-modal-header')
    if (header && !header.querySelector('.headings-modal-search')) {
      const searchWrapper = document.createElement('div')
      searchWrapper.className = 'headings-modal-search'

      const input = document.createElement('input')
      input.type = 'search'
      input.placeholder = 'search headings (/)'
      input.autocomplete = 'off'
      input.spellcheck = false

      input.addEventListener('input', () => {
        searchQuery = input.value
        updateFilteredHeadings()
        currentIndex = 0
        updateActiveItem()
      })

      input.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
          event.stopPropagation()
          if (input.value) {
            input.value = ''
            searchQuery = ''
            updateFilteredHeadings()
            currentIndex = 0
            updateActiveItem()
          } else {
            closeModal()
          }
        } else if (event.key === 'ArrowDown') {
          event.preventDefault()
          currentIndex = Math.min(currentIndex + 1, getVisibleHeadings().length - 1)
          updateActiveItem()
        } else if (event.key === 'ArrowUp') {
          event.preventDefault()
          currentIndex = Math.max(currentIndex - 1, 0)
          updateActiveItem()
        } else if (event.key === 'Enter') {
          event.preventDefault()
          jumpToHeading(currentIndex)
        }
      })

      searchWrapper.appendChild(input)
      header.appendChild(searchWrapper)
      searchInput = input
    } else if (header) {
      searchInput = header.querySelector('.headings-modal-search input') as HTMLInputElement | null
    }

    registerEscapeHandler(modal, closeModal)

    const backdrop = modal.querySelector('.headings-modal-backdrop')
    if (backdrop) {
      backdrop.addEventListener('click', closeModal)
      window.addCleanup(() => backdrop.removeEventListener('click', closeModal))
    }
  }

  document.addEventListener('keydown', handleKeyDown)
  document.addEventListener('keydown', handleGlobalKeyDown)

  window.addCleanup(() => {
    document.removeEventListener('keydown', handleKeyDown)
    document.removeEventListener('keydown', handleGlobalKeyDown)
    clearSecondaryMode()
  })
})

document.addEventListener('nav', () => {
  isOpen = false
  isSecondaryMode = false
  allHeadings = []
  filteredHeadings = []
  searchQuery = ''
  currentIndex = 0
  clearSecondaryMode()

  if (modal) {
    modal.style.display = 'none'
  }
})
