import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import { revealHeading } from './heading-reveal'
import { currentNavSignal } from './nav-lifecycle'

interface HeadingEntry {
  element: HTMLElement
  initial: string | undefined
  row: HTMLAnchorElement
}

// Home row first; j and k stay reserved for movement.
const PICK_KEYS = 'asdflhguiopwertycvbnmxz'.split('')
const CHORD_WINDOW_MS = 1000
const BROWSE_HELP = '{↑↓} {jk} move · {↵} jump · {a–z} first letter · {esc} close'

let dialog: HTMLDialogElement | null = null
let entries: HeadingEntry[] = []
let cursor = 0
let picks = new Map<string, number>()
let help = ''
let pendingG = -Infinity
let ag: RoughAnnotation | null = null
let activeSignal: AbortSignal | undefined

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

function normalizeHeadingText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function visibleHeadingText(node: Node, skipMath = false): string {
  if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? ''
  if (!(node instanceof Element)) return ''
  if (node.getAttribute('aria-hidden') === 'true') return ''
  if (node.tagName.toLowerCase() === 'annotation') return ''
  if (skipMath && node.classList.contains('katex')) return ''

  return Array.from(node.childNodes)
    .map(child => visibleHeadingText(child, skipMath))
    .join('')
}

/** Prose supplies the initial; math does only when the heading has no prose letter, matching the Obsidian picker. */
function headingInitial(heading: HTMLElement, text: string): string | undefined {
  const letter = /[a-z]/i
  return (visibleHeadingText(heading, true).match(letter) ?? text.match(letter))?.[0].toLowerCase()
}

function headingDisplayText(el: Element): string {
  const alias = normalizeHeadingText(el.getAttribute('data-heading-alias') ?? '')
  return alias.length > 0 ? alias : normalizeHeadingText(visibleHeadingText(el))
}

/** Rendered math and code keep their markup; the plain alias would flatten `$\ell_p$` to `ℓp`. */
function headingLabel(heading: HTMLElement, text: string): Node {
  const source = heading.querySelector<HTMLElement>('span.highlight-span') ?? heading
  if (!source.querySelector('.katex, code')) return document.createTextNode(text)

  const clone = source.cloneNode(true) as HTMLElement
  clone
    .querySelectorAll(
      'a[data-role="anchor"], .collapse-rail, .collapsed-dots, button, script, style',
    )
    .forEach(node => node.remove())
  clone.querySelectorAll('[id], [tabindex]').forEach(node => {
    node.removeAttribute('id')
    node.removeAttribute('tabindex')
  })
  // The row is already a link, so links inside the heading collapse to their content.
  clone.querySelectorAll('a').forEach(link => link.replaceWith(...Array.from(link.childNodes)))

  const fragment = document.createDocumentFragment()
  fragment.append(...Array.from(clone.childNodes))
  return fragment
}

function collectHeadings(): HeadingEntry[] {
  const headings = Array.from(
    document.querySelectorAll<HTMLElement>('.page-content :is(h2, h3, h4, h5, h6)[id]'),
  )
    .map(element => ({ element, text: headingDisplayText(element) }))
    .filter(({ text }) => text.length > 0)
  const top = Math.min(...headings.map(({ element }) => Number(element.tagName[1])))

  return headings.map(({ element, text }, index) => {
    const depth = Number(element.tagName[1]) - top
    const url = new URL(window.location.href)
    url.hash = element.id

    const row = document.createElement('a')
    row.className = 'heading-item'
    row.href = url.href
    row.tabIndex = -1
    row.dataset.index = String(index)
    row.dataset.depth = String(depth)
    row.style.setProperty('--depth', String(depth))

    const hint = document.createElement('span')
    hint.className = 'heading-hint'
    hint.setAttribute('aria-hidden', 'true')

    const label = document.createElement('span')
    label.className = 'heading-text'
    label.append(headingLabel(element, text))

    row.append(hint, label)
    return { element, initial: headingInitial(element, text), row }
  })
}

function renderList() {
  const list = dialog?.querySelector('.headings-list')
  if (!list) return

  if (entries.length === 0) {
    const empty = document.createElement('li')
    empty.className = 'heading-empty'
    empty.textContent = 'no headings on this page'
    list.replaceChildren(empty)
  } else {
    list.replaceChildren(
      ...entries.map(entry => {
        const item = document.createElement('li')
        item.append(entry.row)
        return item
      }),
    )
  }
}

/** `{key}` segments render as `<kbd>`; the status region only changes when the text does. */
function setHelp(template: string) {
  const status = dialog?.querySelector('.headings-modal-help')
  if (!status || help === template) return
  help = template
  status.replaceChildren(
    ...template.split(/\{([^}]+)\}/).map((part, index) => {
      if (index % 2 === 0) return part
      const kbd = document.createElement('kbd')
      kbd.textContent = part
      return kbd
    }),
  )
}

/** The section being read is the last heading above the top quarter of the viewport. */
function currentSectionIndex(): number {
  const line = window.innerHeight * 0.25
  let current = -1
  entries.forEach((entry, index) => {
    const rect = entry.element.getBoundingClientRect()
    if (rect.height > 0 && rect.top <= line) current = index
  })
  return current
}

function setCursor(index: number, focus = true) {
  if (entries.length === 0) return
  const previous = entries[cursor]?.row
  cursor = Math.max(0, Math.min(entries.length - 1, index))
  const row = entries[cursor].row

  if (previous && previous !== row) {
    previous.classList.remove('is-active')
    previous.tabIndex = -1
  }
  row.classList.add('is-active')
  row.tabIndex = 0
  if (focus && document.activeElement !== row) row.focus({ preventScroll: true })
  row.scrollIntoView({ block: 'nearest' })
}

function move(delta: number) {
  if (picks.size === 0) setHelp(BROWSE_HELP)
  setCursor(cursor + delta)
}

function resetPick() {
  for (const index of picks.values()) {
    const row = entries[index]?.row
    if (!row) continue
    row.querySelector('.heading-hint')?.replaceChildren()
    row.removeAttribute('aria-keyshortcuts')
  }
  picks = new Map()
  if (dialog) delete dialog.dataset.mode
}

function clearPick() {
  resetPick()
  setHelp(BROWSE_HELP)
}

function chooseInitial(letter: string) {
  const matches = entries.flatMap((entry, index) => (entry.initial === letter ? [index] : []))
  if (matches.length === 0) {
    setHelp(`no heading starts with {${letter}} · {esc} close`)
    return
  }
  if (matches.length === 1) {
    jump(matches[0])
    return
  }

  // Keys go to the matches nearest the cursor, then read top to bottom in list order.
  const marked = [...matches]
    .sort((a, b) => Math.abs(a - cursor) - Math.abs(b - cursor))
    .slice(0, PICK_KEYS.length)
    .sort((a, b) => a - b)
  picks = new Map(marked.map((index, i) => [PICK_KEYS[i], index]))
  for (const [key, index] of picks) {
    const row = entries[index].row
    row.querySelector('.heading-hint')?.replaceChildren(key)
    row.setAttribute('aria-keyshortcuts', key)
  }
  if (dialog) dialog.dataset.mode = 'pick'
  setHelp(`{${letter}} ${matches.length} headings · press the marked key · {esc} back`)
}

function jump(index: number) {
  const entry = entries[index]
  if (!entry || !dialog) return
  dialog.close()
  revealHeading(entry.element)

  ag?.remove()
  const highlight = entry.element.querySelector<HTMLElement>('span.highlight-span')
  if (highlight) {
    const annotation = annotate(highlight, {
      type: 'box',
      color: 'rgba(234, 157, 52, 0.45)',
      animate: false,
      multiline: true,
    })
    ag = annotation
    setTimeout(() => annotation.show(), 50)
    setTimeout(() => annotation.remove(), 2500)
  }

  const rect = entry.element.getBoundingClientRect()
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  window.scrollTo({
    top: window.scrollY + rect.top - window.innerHeight / 2 + rect.height / 2,
    behavior: reduce ? 'auto' : 'smooth',
  })
  history.pushState(null, '', entry.row.href)
}

function openModal() {
  if (!dialog || dialog.open) return
  if (document.documentElement.getAttribute('reader-mode') === 'on') return

  entries = collectHeadings()
  cursor = 0
  renderList()
  setHelp(entries.length === 0 ? '{esc} close' : BROWSE_HELP)
  dialog.showModal()

  if (entries.length === 0) {
    dialog.querySelector<HTMLElement>('.headings-modal-close')?.focus()
    return
  }
  const here = currentSectionIndex()
  entries[here]?.row.setAttribute('aria-current', 'location')
  setCursor(Math.max(here, 0))
}

function onDialogKeyDown(event: KeyboardEvent) {
  if (event.isComposing || event.metaKey || event.altKey) return
  const { key } = event

  if (event.ctrlKey) {
    if (key === 'n') move(1)
    else if (key === 'p') move(-1)
    else return
  } else if (key === 'Escape') {
    if (picks.size > 0) clearPick()
    else dialog?.close()
  } else if (key === 'ArrowDown' || key === 'j') {
    move(1)
  } else if (key === 'ArrowUp' || key === 'k') {
    move(-1)
  } else if (key === 'Home') {
    move(-entries.length)
  } else if (key === 'End') {
    move(entries.length)
  } else if (key === 'Enter') {
    if (event.target instanceof Element && event.target.closest('.headings-modal-close')) return
    jump(cursor)
  } else if (/^[a-z]$/i.test(key)) {
    // Shift reaches the j and k initials that movement reserves.
    const letter = key.toLowerCase()
    if (picks.size === 0) chooseInitial(letter)
    else if (picks.has(letter)) jump(picks.get(letter)!)
  } else {
    return
  }

  event.preventDefault()
  event.stopPropagation()
}

function onDocumentKeyDown(event: KeyboardEvent) {
  if (dialog?.open) return
  if (event.ctrlKey || event.metaKey || event.altKey || shouldIgnoreShortcutTarget(event.target)) {
    pendingG = -Infinity
    return
  }
  if (event.key === 'h' && event.timeStamp - pendingG < CHORD_WINDOW_MS) {
    event.preventDefault()
    pendingG = -Infinity
    openModal()
    return
  }
  pendingG = event.key === 'g' ? event.timeStamp : -Infinity
}

function rowIndex(target: EventTarget | null): number | undefined {
  const row = target instanceof Element ? target.closest<HTMLElement>('.heading-item') : null
  return row ? Number(row.dataset.index) : undefined
}

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (activeSignal === signal) return
  activeSignal = signal

  dialog = document.querySelector<HTMLDialogElement>('dialog.headings-modal')
  entries = []
  picks = new Map()
  help = ''
  pendingG = -Infinity

  document.addEventListener('keydown', onDocumentKeyDown, { signal })
  if (!dialog) return
  const modal = dialog

  modal.addEventListener('keydown', onDialogKeyDown, { signal })
  modal.addEventListener('close', resetPick, { signal })
  modal
    .querySelector('.headings-modal-close')
    ?.addEventListener('click', () => modal.close(), { signal })

  // Clicks on the backdrop target the dialog itself, outside its box.
  modal.addEventListener(
    'click',
    event => {
      if (event.target !== modal) return
      const rect = modal.getBoundingClientRect()
      const inside =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom
      if (!inside) modal.close()
    },
    { signal },
  )

  const list = modal.querySelector('.headings-list')
  list?.addEventListener(
    'click',
    event => {
      const index = rowIndex(event.target)
      const mouse = event as MouseEvent
      if (index === undefined || mouse.button !== 0) return
      if (mouse.metaKey || mouse.ctrlKey || mouse.shiftKey || mouse.altKey) return
      event.preventDefault()
      jump(index)
    },
    { signal },
  )
  list?.addEventListener(
    'focusin',
    event => {
      const index = rowIndex(event.target)
      if (index !== undefined) setCursor(index, false)
    },
    { signal },
  )
})
