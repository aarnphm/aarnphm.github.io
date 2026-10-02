import { Component } from 'obsidian'

interface MarginNote {
  wrapper: HTMLElement
  content: HTMLElement
  label: HTMLButtonElement
  pane: HTMLElement
  column: HTMLElement
  leftSpace: number
  rightSpace: number
}

export class SidenoteLayout extends Component {
  private frame = 0
  private observer?: MutationObserver

  constructor(private root: HTMLElement) {
    super()
  }

  onload(): void {
    this.observer = new MutationObserver(() => this.schedule())
    this.observer.observe(this.root, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['hidden', 'data-sidenote-open'],
    })
    this.schedule()
  }

  onunload(): void {
    this.observer?.disconnect()
    cancelAnimationFrame(this.frame)
    for (const wrapper of Array.from(
      this.root.querySelectorAll<HTMLElement>('.sidenote[data-placement]'),
    )) {
      delete wrapper.dataset.placement
      delete wrapper.dataset.side
      const content = wrapper.querySelector<HTMLElement>('.sidenote-content')
      content?.style.removeProperty('width')
      content?.style.removeProperty('left')
      content?.style.removeProperty('top')
    }
  }

  schedule(): void {
    if (this.frame) return
    this.frame = requestAnimationFrame(() => {
      this.frame = 0
      this.layout()
    })
  }

  private layout(): void {
    const notes: MarginNote[] = []
    for (const wrapper of Array.from(
      this.root.querySelectorAll<HTMLElement>('.garden-plugin-ui.sidenote'),
    )) {
      const label = wrapper.querySelector<HTMLButtonElement>('.sidenote-label')
      const content = wrapper.querySelector<HTMLElement>('.sidenote-content')
      const pane = wrapper.closest<HTMLElement>('.markdown-preview-view, .cm-scroller')
      const column = wrapper.closest<HTMLElement>('.cm-line, .markdown-preview-sizer > div')
      if (!label || !content || !pane || !column || !pane.getBoundingClientRect().width) continue
      const paneRect = pane.getBoundingClientRect()
      const columnRect = column.getBoundingClientRect()
      const gap = 16
      const right = paneRect.right - columnRect.right - gap * 2
      const left = columnRect.left - paneRect.left - gap * 2
      const inline = content.classList.contains('sidenote-inline')
      const leftSpace = wrapper.dataset.allowLeft !== 'false' ? left : 0
      const rightSpace = wrapper.dataset.allowRight !== 'false' ? right : 0
      const margin = !inline && Math.max(leftSpace, rightSpace) >= 160
      if (wrapper.classList.contains('sidenote-source') && wrapper.hidden === margin)
        wrapper.hidden = !margin
      wrapper.dataset.placement = margin ? 'margin' : 'inline'
      const open =
        wrapper.dataset.sidenoteOpen === undefined
          ? margin
          : wrapper.dataset.sidenoteOpen === 'true'
      if (content.hidden === open) content.hidden = !open
      wrapper.classList.toggle('open', open)
      label.setAttribute('aria-expanded', String(open))
      if (!margin) {
        delete wrapper.dataset.side
        content.style.removeProperty('width')
        content.style.removeProperty('left')
        content.style.removeProperty('top')
        continue
      }
      notes.push({ wrapper, content, label, pane, column, leftSpace, rightSpace })
    }

    const bottoms = new Map<HTMLElement, { left: number; right: number }>()
    for (const note of notes.sort(
      (a, b) => a.label.getBoundingClientRect().top - b.label.getBoundingClientRect().top,
    )) {
      const labelRect = note.label.getBoundingClientRect()
      const columnRect = note.column.getBoundingClientRect()
      const previous = bottoms.get(note.pane) ?? { left: -Infinity, right: -Infinity }
      // Each pane balances its margins independently, keeping explicit side restrictions.
      const side =
        note.leftSpace >= 160 && (note.rightSpace < 160 || previous.left <= previous.right)
          ? 'left'
          : 'right'
      const width = Math.min(272, side === 'left' ? note.leftSpace : note.rightSpace)
      const target = side === 'right' ? columnRect.right + 16 : columnRect.left - 16 - width
      note.wrapper.dataset.side = side
      note.content.style.width = `${width}px`
      note.content.style.left = `${target - labelRect.left}px`
      const top = Math.max(labelRect.top, previous[side] + 8)
      note.content.style.top = `${top - labelRect.top}px`
      if (!note.content.hidden) previous[side] = top + note.content.getBoundingClientRect().height
      bottoms.set(note.pane, previous)
    }
  }
}
