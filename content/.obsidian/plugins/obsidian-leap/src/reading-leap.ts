import { App, Component, MarkdownView } from 'obsidian'
import type { CharacterMotion, LeapOptions } from './types'
import { characterLabelAlphabet } from './character-targets'
import { LeapOverlay } from './overlay'

const equivalents = [' \t\r\n', '([{', ')]}', '\'"`']
const excluded =
  '.sidenote-content, .garden-leap-overlay, button, input, textarea, select, [contenteditable]:not([contenteditable="false"]), script, style, svg, .copy-code-button'
const semanticElements =
  'a, strong, em, s, del, mark, sub, sup, code, pre, p, h1, h2, h3, h4, h5, h6, li, ul, ol, blockquote, table, thead, tbody, tr, th, td, figure, figcaption, dl, dt, dd, section, article, .callout, .markdown-preview-section'
const blockElements =
  'pre, p, h1, h2, h3, h4, h5, h6, li, ul, ol, blockquote, table, thead, tbody, tr, th, td, figure, figcaption, dl, dt, dd, section, article, .callout, .markdown-preview-section'

interface ReadingPane {
  view: MarkdownView
  pane: HTMLElement
}

interface Caret {
  node: Text
  offset: number
}

interface CharacterTarget {
  range: Range
  caret: Caret
  rect: DOMRect
}

interface QuerySession extends ReadingPane {
  kind: 'query'
  motion: CharacterMotion
  origin: Range
}

interface CharacterSession extends ReadingPane {
  kind: 'character'
  motion: CharacterMotion
  targets: CharacterTarget[]
  index: number
}

interface SyntaxSession extends ReadingPane {
  kind: 'syntax'
  linewise: boolean
  targets: Range[]
  index: number
}

type Session = QuerySession | CharacterSession | SyntaxSession

function isText(node: Node): node is Text {
  return node.nodeType === Node.TEXT_NODE
}

function intersects(rect: DOMRect, clip: DOMRect): boolean {
  return (
    rect.height > 0 &&
    rect.bottom > clip.top &&
    rect.top < clip.bottom &&
    rect.right >= clip.left &&
    rect.left < clip.right
  )
}

function textNodes(pane: HTMLElement): Text[] {
  const nodes: Text[] = []
  const walker = pane.ownerDocument.createTreeWalker(pane, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!isText(node) || !node.data || node.parentElement?.closest(excluded)) continue
    const parent = node.parentElement
    if (!parent || !parent.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }))
      continue
    nodes.push(node)
  }
  return nodes
}

function characterRange(node: Text, from: number, to: number): Range {
  const range = node.ownerDocument.createRange()
  range.setStart(node, from)
  range.setEnd(node, to)
  return range
}

function precedingCaret(nodes: Text[], nodeIndex: number, offset: number): Caret {
  const node = nodes[nodeIndex]
  if (offset > 0) {
    const previous = Array.from(node.data.slice(Math.max(0, offset - 2), offset)).at(-1)
    return { node, offset: offset - (previous?.length ?? 0) }
  }
  const previous = nodes[nodeIndex - 1]
  if (!previous) return { node, offset: 0 }
  const character = Array.from(previous.data.slice(-2)).at(-1)
  return { node: previous, offset: previous.length - (character?.length ?? 0) }
}

function sameRange(left: Range, right: Range): boolean {
  return (
    left.compareBoundaryPoints(Range.START_TO_START, right) === 0 &&
    left.compareBoundaryPoints(Range.END_TO_END, right) === 0
  )
}

export class ReadingLeap extends Component {
  private session?: Session
  private prefix?: { view: MarkdownView; time: number }
  private lastCharacter?: string
  private overlay?: LeapOverlay
  private observer?: MutationObserver
  private expectedScroll?: { pane: HTMLElement; top: number; left: number }

  constructor(
    private app: App,
    private options: () => LeapOptions,
  ) {
    super()
  }

  onload(): void {
    const document = this.app.workspace.containerEl.ownerDocument
    this.registerDomEvent(document, 'keydown', event => this.handleKey(event), { capture: true })
    this.registerDomEvent(document, 'scroll', event => this.handleScroll(event), { capture: true })
    this.registerDomEvent(document, 'pointerdown', () => this.cancel(), { capture: true })
    this.registerEvent(this.app.workspace.on('active-leaf-change', () => this.cancel()))
    this.registerEvent(this.app.workspace.on('file-open', () => this.cancel()))
    this.registerEvent(this.app.workspace.on('layout-change', () => this.cancel()))
    this.registerEvent(this.app.workspace.on('resize', () => this.cancel()))
    this.register(() => this.cancel())
  }

  startCharacter(motion: CharacterMotion): boolean {
    const reading = this.readingPane()
    if (!reading) return false
    this.cancel()
    const origin = this.originRange(reading.pane, motion === 'F' || motion === 'T')
    this.begin({ ...reading, kind: 'query', motion, origin })
    return true
  }

  startSyntax(linewise: boolean): boolean {
    const reading = this.readingPane()
    if (!reading) return false
    this.cancel()
    const selection = reading.pane.ownerDocument.getSelection()
    const selected = selection?.rangeCount ? selection.getRangeAt(0) : undefined
    const origin =
      selected &&
      reading.pane.contains(selected.startContainer) &&
      reading.pane.contains(selected.endContainer)
        ? selected.cloneRange()
        : this.originRange(reading.pane, false)
    let node: Node | null = origin.commonAncestorContainer
    if (node === reading.pane) {
      node = this.firstVisibleText(reading.pane) ?? null
    }
    let element = node instanceof Element ? node : node?.parentElement
    const targets: Range[] = []
    while (element && element !== reading.pane) {
      if (element.matches(linewise ? blockElements : semanticElements)) {
        const nodes = textNodes(element instanceof HTMLElement ? element : reading.pane)
        const first = nodes[0]
        const last = nodes.at(-1)
        if (first && last) {
          const range = characterRange(first, 0, first.length)
          range.setEnd(last, last.length)
          if (!targets.some(target => sameRange(target, range))) targets.push(range)
        }
      }
      element = element.parentElement
    }
    if (!targets.length) return false
    this.begin({ ...reading, kind: 'syntax', linewise, targets, index: 0 })
    this.selectSyntax()
    return true
  }

  cancel(): void {
    this.session = undefined
    this.prefix = undefined
    this.expectedScroll = undefined
    this.observer?.disconnect()
    this.observer = undefined
    this.overlay?.clear()
    this.overlay = undefined
  }

  private readingPane(): ReadingPane | undefined {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView)
    if (!view || view.getMode() !== 'preview') return undefined
    const container = view.previewMode.containerEl
    const pane = container.matches('.markdown-preview-view')
      ? container
      : container.querySelector<HTMLElement>('.markdown-preview-view')
    return pane ? { view, pane } : undefined
  }

  private originRange(pane: HTMLElement, backward: boolean): Range {
    const selection = pane.ownerDocument.getSelection()
    const node = selection?.focusNode
    const origin = pane.ownerDocument.createRange()
    if (
      node &&
      pane.contains(node) &&
      !(node instanceof Element ? node : node.parentElement)?.closest(excluded)
    ) {
      origin.setStart(node, selection.focusOffset)
    } else {
      origin.setStart(pane, backward ? pane.childNodes.length : 0)
    }
    origin.collapse(true)
    return origin
  }

  private firstVisibleText(pane: HTMLElement): Text | undefined {
    const clip = pane.getBoundingClientRect()
    return textNodes(pane).find(node =>
      Array.from(characterRange(node, 0, node.length).getClientRects()).some(rect =>
        intersects(rect, clip),
      ),
    )
  }

  private begin(session: Session): void {
    this.session = session
    this.overlay = new LeapOverlay(session.pane.ownerDocument)
    this.observer = new MutationObserver(() => this.cancel())
    this.observer.observe(session.pane, { childList: true, characterData: true, subtree: true })
  }

  private canHandle(event: KeyboardEvent, reading: ReadingPane): boolean {
    const target = event.target
    if (
      !(target instanceof Element) ||
      target.closest(`${excluded}, .modal-container, [role="button"], .clickable-icon`)
    ) {
      return false
    }
    const modal = reading.pane.ownerDocument.querySelector<HTMLElement>('.modal-container .modal')
    if (modal?.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return false
    return reading.pane.contains(target) || target.contains(reading.pane)
  }

  private handleKey(event: KeyboardEvent): void {
    if (event.key === 'Shift') return
    if (event.defaultPrevented || event.repeat || event.isComposing) return
    const reading = this.readingPane()
    if (
      event.ctrlKey ||
      event.altKey ||
      event.metaKey ||
      !reading ||
      !this.canHandle(event, reading)
    ) {
      this.cancel()
      return
    }
    if (this.session) {
      if (this.session.view !== reading.view || this.session.pane !== reading.pane) {
        this.cancel()
        return
      }
      if (this.handleSessionKey(event)) this.consume(event)
      return
    }
    if (!this.options().readingMotions) {
      this.prefix = undefined
      return
    }
    const prefix = this.prefix
    this.prefix = undefined
    if (event.key === 'g') {
      this.prefix = { view: reading.view, time: performance.now() }
      return
    }
    if (
      (event.key === 'a' || event.key === 'A') &&
      prefix?.view === reading.view &&
      performance.now() - prefix.time < 1000
    ) {
      if (this.startSyntax(event.key === 'A')) this.consume(event)
      return
    }
    if (event.key === 'f' || event.key === 'F' || event.key === 't' || event.key === 'T') {
      if (this.startCharacter(event.key)) this.consume(event)
    }
  }

  private handleSessionKey(event: KeyboardEvent): boolean {
    const session = this.session
    if (!session) return false
    if (event.key === 'Escape') {
      this.cancel()
      return true
    }
    if (session.kind === 'query') {
      const character = event.key === 'Enter' ? this.lastCharacter : event.key
      if (!character || Array.from(character).length !== 1) {
        this.cancel()
        return event.key === 'Enter'
      }
      this.lastCharacter = character
      const targets = this.characterTargets(session, character)
      if (!targets.length) {
        this.cancel()
        return true
      }
      this.session = { ...session, kind: 'character', targets, index: 0 }
      this.jumpCharacter()
      return true
    }
    if (session.kind === 'syntax') {
      const next = event.key === 'Enter' || (!session.linewise && event.key === 'a')
      const previous = event.key === 'Backspace' || (!session.linewise && event.key === 'A')
      if (next || previous) {
        session.index =
          previous && session.index === 0
            ? session.targets.length - 1
            : Math.min(session.targets.length - 1, session.index + (next ? 1 : -1))
        this.selectSyntax()
        return true
      }
    } else {
      const forward = session.motion.toLowerCase()
      const backward = forward.toUpperCase()
      const next = event.key === forward || event.key === 'Enter' || event.key === ' '
      const previous = event.key === backward || event.key === 'Backspace'
      if (next || previous) {
        session.index =
          previous && session.index === 0
            ? session.targets.length - 1
            : Math.min(session.targets.length - 1, session.index + (next ? 1 : -1))
        this.jumpCharacter()
        return true
      }
      if (this.options().showLabels) {
        const label = characterLabelAlphabet(session.motion).indexOf(event.key)
        const index = label + 1
        if (label >= 0 && index < session.targets.length) {
          session.index = index
          this.jumpCharacter()
          this.cancel()
          return true
        }
      }
    }
    this.cancel()
    return false
  }

  private characterTargets(session: QuerySession, character: string): CharacterTarget[] {
    const group = equivalents.find(value => value.includes(character)) ?? character
    const backward = session.motion === 'F' || session.motion === 'T'
    const till = session.motion === 't' || session.motion === 'T'
    const nodes = textNodes(session.pane)
    const clip = session.pane.getBoundingClientRect()
    const targets: CharacterTarget[] = []
    for (const [nodeIndex, node] of nodes.entries()) {
      const nodeRange = characterRange(node, 0, node.length)
      if (!Array.from(nodeRange.getClientRects()).some(rect => intersects(rect, clip))) continue
      for (let from = 0; from < node.length; ) {
        const point = node.data.codePointAt(from)
        if (point === undefined) break
        const value = String.fromCodePoint(point)
        const to = from + value.length
        const position = from
        from = to
        if (!group.includes(value)) continue
        const previous = Array.from(node.data.slice(Math.max(0, position - 2), position)).at(-1)
        const following = node.data.codePointAt(to)
        if (
          previous &&
          group.includes(previous) &&
          following !== undefined &&
          group.includes(String.fromCodePoint(following))
        ) {
          continue
        }
        const range = characterRange(node, position, to)
        const comparison = range.compareBoundaryPoints(Range.START_TO_START, session.origin)
        if (backward ? comparison >= 0 : comparison <= 0) continue
        const rect = Array.from(range.getClientRects()).find(candidate =>
          intersects(candidate, clip),
        )
        if (!rect) continue
        const caret = till
          ? backward
            ? { node, offset: to }
            : precedingCaret(nodes, nodeIndex, position)
          : { node, offset: position }
        const cursor = characterRange(caret.node, caret.offset, caret.offset)
        if (cursor.compareBoundaryPoints(Range.START_TO_START, session.origin) === 0) continue
        targets.push({ range, caret, rect })
      }
    }
    return backward ? targets.reverse() : targets
  }

  private jumpCharacter(): void {
    const session = this.session
    if (session?.kind !== 'character') return
    const target = session.targets[session.index]
    if (!target?.caret.node.isConnected) {
      this.cancel()
      return
    }
    const selection = session.pane.ownerDocument.getSelection()
    selection?.collapse(target.caret.node, target.caret.offset)
    const caretRect = characterRange(
      target.caret.node,
      target.caret.offset,
      target.caret.offset,
    ).getClientRects()[0]
    if (caretRect) this.keepVisible(session.pane, caretRect)
    if (session.targets.length === 1) {
      this.cancel()
      return
    }
    const labels = characterLabelAlphabet(session.motion)
    this.overlay?.show(
      session.targets.map((candidate, index) => ({
        rect: candidate.rect,
        active: index === session.index,
        label: this.options().showLabels && index > session.index ? labels[index - 1] : undefined,
      })),
    )
  }

  private selectSyntax(): void {
    const session = this.session
    if (session?.kind !== 'syntax') return
    const range = session.targets[session.index]
    if (!range?.startContainer.isConnected || !range.endContainer.isConnected) {
      this.cancel()
      return
    }
    const selection = session.pane.ownerDocument.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range.cloneRange())
    if (session.targets.length === 1) {
      this.cancel()
      return
    }
    const clip = session.pane.getBoundingClientRect()
    this.overlay?.show(
      Array.from(range.getClientRects())
        .filter(rect => intersects(rect, clip))
        .map(rect => ({ rect, active: true })),
    )
  }

  private keepVisible(pane: HTMLElement, rect: DOMRect): void {
    const clip = pane.getBoundingClientRect()
    const distance =
      rect.top < clip.top ? rect.top - clip.top : Math.max(0, rect.bottom - clip.bottom)
    if (!distance) return
    pane.scrollTop += distance
    this.expectedScroll = { pane, top: pane.scrollTop, left: pane.scrollLeft }
  }

  private handleScroll(event: Event): void {
    const session = this.session
    const target = event.target
    if (!session || !(target instanceof Node)) return
    if (target !== session.pane && !session.pane.contains(target) && !target.contains(session.pane))
      return
    const expected = this.expectedScroll
    this.expectedScroll = undefined
    if (
      expected?.pane === session.pane &&
      expected.top === session.pane.scrollTop &&
      expected.left === session.pane.scrollLeft
    ) {
      return
    }
    this.cancel()
  }

  private consume(event: KeyboardEvent): void {
    event.preventDefault()
    event.stopPropagation()
  }
}
