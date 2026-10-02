import {
  App,
  Component,
  MarkdownRenderer,
  MarkdownView,
  Modal,
  Notice,
  Plugin,
  type Modifier,
} from 'obsidian'

type HeadingItem = { line: number; level: number; text: string }

const LETTER_KEYS = 'abcdefghijklmnopqrstuvwxyz'.split('')
const SECONDARY_KEYS = [
  'a',
  's',
  'd',
  'f',
  'l',
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

function parseHeadings(text: string): HeadingItem[] {
  const lines = text.split(/\r?\n/)
  const results: HeadingItem[] = []
  let inFrontmatter = false
  let checkedFirstContent = false
  let fence: { marker: string; length: number } | undefined
  let displayMath = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    if (!checkedFirstContent && trimmed !== '') {
      checkedFirstContent = true
      if (trimmed === '---') {
        inFrontmatter = true
        continue
      }
    }

    if (inFrontmatter) {
      if (trimmed === '---' || trimmed === '...') {
        inFrontmatter = false
      }
      continue
    }

    const fenceMatch = /^ {0,3}(`{3,}|~{3,})/.exec(line)
    if (fence) {
      if (
        fenceMatch &&
        fenceMatch[1][0] === fence.marker &&
        fenceMatch[1].length >= fence.length &&
        trimmed.slice(fenceMatch[1].length).trim() === ''
      )
        fence = undefined
      continue
    }
    if (fenceMatch) {
      fence = { marker: fenceMatch[1][0], length: fenceMatch[1].length }
      continue
    }

    // A `=` or `-` line inside `$$` display math would otherwise read as a Setext underline.
    const mathDelimiters = line.match(/\$\$/g)?.length ?? 0
    if (displayMath || mathDelimiters % 2 === 1) {
      if (mathDelimiters % 2 === 1) displayMath = !displayMath
      continue
    }

    const atx = line.match(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/)
    if (atx) {
      const [, hashes, title] = atx
      results.push({ line: i, level: hashes.length, text: title.trim() })
      continue
    }

    if (i + 1 < lines.length && trimmed !== '') {
      const underline = lines[i + 1]
      if (/^\s{0,3}=+\s*$/.test(underline)) {
        results.push({ line: i, level: 1, text: line.trim() })
      } else if (/^\s{0,3}-+\s*$/.test(underline)) {
        results.push({ line: i, level: 2, text: line.trim() })
      }
    }
  }

  return results
}

function collectInitials(text: string): string[] {
  const initials: string[] = []
  const seen = new Set<string>()
  const add = (raw?: string) => {
    if (!raw) return
    const ch = raw.match(/[A-Za-z]/)?.[0]?.toLowerCase()
    if (!ch || seen.has(ch)) return
    seen.add(ch)
    initials.push(ch)
  }

  const trimmed = text.trim()
  const link = trimmed.match(/^\[\[([^[\]]+)\]\]/)
  if (link) {
    const inner = link[1]
    const [target, alias] = inner.split('|')
    add(target)
    target?.split('/').forEach(add)
    add(alias)
  } else {
    // Prose supplies the initial; inline math does only when the heading has no prose letter.
    add(trimmed.replace(/\$[^$]*\$/g, ' '))
    if (initials.length === 0) add(trimmed)
  }

  return initials
}

class HeadingNavigatorModal extends Modal {
  private renderComponent = new Component()
  private headings: HeadingItem[]
  private itemEls: HTMLButtonElement[] = []
  private hintEls: HTMLSpanElement[] = []
  private groups = new Map<string, number[]>()
  private secondaryMap = new Map<string, number>()
  private cursor = 0
  private secondaryActive = false
  private listEl: HTMLDivElement | null = null
  private visibleIndices: number[] = []

  constructor(
    app: App,
    private view: MarkdownView,
    headings: HeadingItem[],
  ) {
    super(app)
    this.headings = headings
    this.visibleIndices = headings.map((_, idx) => idx)
    this.buildGroups()
  }

  onOpen(): void {
    this.renderComponent.load()
    this.modalEl.addClass('heading-gh-modal')
    this.setTitle('Jump to heading')
    const container = this.modalEl.closest('.modal-container')
    container?.addClass('heading-gh-container')
    this.contentEl.addClass('heading-gh-body')
    this.renderList()
    this.bindKeys()
    this.highlight(this.cursor)
  }

  onClose(): void {
    this.renderComponent.unload()
    this.contentEl.empty()
  }

  private buildGroups() {
    this.visibleIndices.forEach(idx => {
      const h = this.headings[idx]
      for (const init of collectInitials(h.text)) {
        const existing = this.groups.get(init) ?? []
        existing.push(idx)
        this.groups.set(init, existing)
      }
    })
  }

  private renderList() {
    const list = this.contentEl.createEl('div', { cls: 'heading-gh-list' })
    this.listEl = list
    // Depth counts from the note's shallowest heading, so an h2-only note starts flush.
    const top = Math.min(...this.headings.map(h => h.level))
    this.headings.forEach((h, idx) => {
      const item = list.createEl('button', {
        cls: 'heading-gh-item',
        attr: { type: 'button', 'data-level': `${h.level}` },
      })
      item.style.setProperty('--heading-depth', `${h.level - top}`)
      item.addEventListener('click', () => this.jump(idx))
      item.addEventListener('focus', () => {
        this.cursor = idx
        this.highlight(idx)
      })

      const hint = item.createEl('span', { cls: 'heading-gh-hint' })
      const label = item.createEl('span', { cls: 'heading-gh-text' })

      this.itemEls[idx] = item
      this.hintEls[idx] = hint
      label.title = h.text
      void this.renderHeading(h, label).catch(error => {
        if (label.isConnected) label.setText(h.text)
        console.error('Heading rendering failed', error)
      })
    })
  }

  private async renderHeading(heading: HeadingItem, label: HTMLElement): Promise<void> {
    await MarkdownRenderer.render(
      this.app,
      heading.text,
      label,
      this.view.file?.path ?? '',
      this.renderComponent,
    )
    if (!label.isConnected) return
    // The row jumps to the heading; rendered links contribute their text without nested actions.
    for (const link of Array.from(label.querySelectorAll('a')))
      link.replaceWith(...Array.from(link.childNodes))
  }

  private bindKeys() {
    const scope = this.scope

    const register = (mods: Modifier[], key: string, fn: () => void) => {
      scope.register(mods, key, evt => {
        evt?.preventDefault()
        evt?.stopPropagation()
        fn()
        return false
      })
    }

    register([], 'j', () => this.move(1))
    register([], 'k', () => this.move(-1))
    register([], 'ArrowDown', () => this.move(1))
    register([], 'ArrowUp', () => this.move(-1))
    register(['Ctrl'], 'n', () => this.move(1))
    register(['Ctrl'], 'p', () => this.move(-1))
    register([], 'Enter', () => this.jump(this.cursor))
    register([], 'q', () => this.close())
    register([], 'Escape', () => {
      if (this.secondaryActive) {
        this.clearSecondary()
      } else {
        this.close()
      }
    })

    LETTER_KEYS.forEach(ch => {
      register([], ch, () => {
        if (this.secondaryActive) {
          this.useSecondary(ch)
        } else {
          this.handleInitial(ch)
        }
      })
    })
  }

  private move(delta: number) {
    if (this.visibleIndices.length === 0) return
    const pos = Math.max(0, this.visibleIndices.indexOf(this.cursor))
    const nextPos = Math.max(0, Math.min(this.visibleIndices.length - 1, pos + delta))
    const target = this.visibleIndices[nextPos]
    if (target === undefined) return
    this.cursor = target
    this.highlight(target)
  }

  private highlight(idx: number) {
    if (!this.visibleIndices.includes(idx)) return
    this.itemEls.forEach(el => el?.removeClass('is-active'))
    const item = this.itemEls[idx]
    item?.addClass('is-active')
    if (item && document.activeElement !== item) item.focus({ preventScroll: true })
    item?.scrollIntoView({ block: 'nearest' })
  }

  private jump(idx: number) {
    if (idx < 0) return
    const target = this.headings[idx]
    if (!target) return
    this.close()

    if (this.view.getMode() === 'preview') {
      this.view.setEphemeralState({ line: target.line, focus: true })
      return
    }

    const editor = this.view.editor
    this.app.workspace.setActiveLeaf(this.view.leaf, { focus: true })
    editor.setCursor({ line: target.line, ch: 0 })
    if (typeof editor.scrollIntoView === 'function') {
      editor.scrollIntoView(
        { from: { line: target.line, ch: 0 }, to: { line: target.line, ch: 0 } },
        true,
      )
    }
    editor.focus()
  }

  private handleInitial(ch: string) {
    const list = this.groups.get(ch)
    if (!list || list.length === 0) return
    if (list.length === 1) {
      this.jump(list[0])
    } else {
      this.enterSecondary(list)
    }
  }

  private enterSecondary(indices: number[]) {
    this.secondaryActive = true
    this.secondaryMap.clear()
    this.clearSecondaryHints()

    const positions = new Map<number, number>()
    this.visibleIndices.forEach((idx, i) => positions.set(idx, i))
    const mid = this.visibleIndices.length / 2

    indices
      .slice(0, SECONDARY_KEYS.length)
      .sort((a, b) => {
        const pa = positions.get(a) ?? 0
        const pb = positions.get(b) ?? 0
        const da = Math.abs(pa - mid)
        const db = Math.abs(pb - mid)
        return da === db ? pa - pb : da - db
      })
      .forEach((idx, i) => {
        const key = SECONDARY_KEYS[i]
        this.secondaryMap.set(key, idx)
        this.hintEls[idx]?.setText(`${key}`)
        this.itemEls[idx]?.addClass('is-candidate')
      })
    this.modalEl.addClass('is-picking')
  }

  private useSecondary(ch: string) {
    const idx = this.secondaryMap.get(ch)
    if (idx !== undefined) {
      this.jump(idx)
    }
  }

  private clearSecondary() {
    this.secondaryActive = false
    this.secondaryMap.clear()
    this.clearSecondaryHints()
  }

  private clearSecondaryHints() {
    this.hintEls.forEach(el => el?.setText(''))
    this.itemEls.forEach(el => el?.removeClass('is-candidate'))
    this.modalEl.removeClass('is-picking')
  }
}

export default class HeadingNavigatorPlugin extends Plugin {
  private navigator?: HeadingNavigatorModal
  private readingPrefix?: { view: MarkdownView; time: number }

  async onload() {
    this.addCommand({
      id: 'heading-gh-navigator',
      name: 'Jump to heading (gh)',
      callback: () => this.openNavigator(),
    })
    this.registerDomEvent(document, 'keydown', event => this.handleReadingKey(event))
    this.registerEvent(
      this.app.workspace.on('active-leaf-change', () => {
        this.readingPrefix = undefined
      }),
    )
    this.registerEvent(
      this.app.workspace.on('layout-change', () => {
        this.readingPrefix = undefined
      }),
    )
  }

  onunload(): void {
    this.navigator?.close()
    this.navigator = undefined
    this.readingPrefix = undefined
  }

  private handleReadingKey(event: KeyboardEvent): void {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView)
    const target = event.target
    if (
      event.defaultPrevented ||
      event.repeat ||
      event.isComposing ||
      event.ctrlKey ||
      event.altKey ||
      event.metaKey ||
      event.shiftKey ||
      !view ||
      view.getMode() !== 'preview' ||
      !(target instanceof HTMLElement) ||
      !view.previewMode.containerEl.contains(target) ||
      target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])')
    ) {
      this.readingPrefix = undefined
      return
    }

    const prefix = this.readingPrefix
    this.readingPrefix = undefined
    if (event.key === 'g') {
      this.readingPrefix = { view, time: performance.now() }
    } else if (
      event.key === 'h' &&
      prefix?.view === view &&
      performance.now() - prefix.time < 1000
    ) {
      event.preventDefault()
      event.stopPropagation()
      this.openNavigator()
    }
  }

  private openNavigator() {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView)
    if (!view) {
      new Notice('Open a Markdown file to jump to headings.')
      return
    }

    const headings = parseHeadings(view.getViewData())
    if (headings.length === 0) {
      new Notice('No headings found in this note.')
      return
    }

    this.navigator?.close()
    this.navigator = new HeadingNavigatorModal(this.app, view, headings)
    this.navigator.open()
  }
}
