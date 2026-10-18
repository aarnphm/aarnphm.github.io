// Camera buttons and, under them, a right-aligned column of topic tabs or tag toggles, as the
// garden's: a collapsed button shows its shortcut, a chosen one opens leftward to "name count".
// A two-segment switch at the column's head picks topics or tags. Note text reaches the buttons
// as text content only.
import { Component } from 'obsidian'
import type { FilterPanel, ViewFilter } from './covers'
import type { TopicTab } from './topics'

export type CameraAction = 'in' | 'out' | 'fit' | 'fullscreen'

export interface PanelHost {
  camera(action: CameraAction): void
  topic(id: string): void
  tag(id: string): void
  panel(tab: FilterPanel): void
}

const ICONS: Record<CameraAction, string> = {
  in: 'M12 5v14M5 12h14',
  out: 'M5 12h14',
  fit: 'M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5',
  fullscreen: 'M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7',
}
const EXIT_ICON = 'M4 14h6v6M3 21l7-7M20 10h-6V4M21 3l-7 7'

const LABELS: Record<CameraAction, string> = {
  in: 'Zoom in',
  out: 'Zoom out',
  fit: 'Fit bookcase',
  fullscreen: 'Full screen',
}

const PANELS: FilterPanel[] = ['topics', 'tags']

let views = 0

function icon(parent: HTMLElement, d: string): SVGPathElement {
  const svg = parent.createSvg('svg', {
    attr: { viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
  })
  return svg.createSvg('path', { attr: { d } })
}

/** Arrow keys, Home and End along a column or row: the index they lead to from `i`, or null. */
function step(key: string, i: number, n: number): number | null {
  if (key === 'ArrowRight' || key === 'ArrowDown') return (i + 1) % n
  if (key === 'ArrowLeft' || key === 'ArrowUp') return (i - 1 + n) % n
  if (key === 'Home') return 0
  if (key === 'End') return n - 1
  return null
}

/** The next button after `i` whose shortcut starts with the typed letter, or null. */
function letter(key: string, i: number, buttons: HTMLButtonElement[]): number | null {
  if (key.length !== 1) return null
  const n = buttons.length
  for (let d = 1; d <= n; d++) {
    const j = (i + d) % n
    if (buttons[j].dataset.shortcut === key.toLowerCase()) return j
  }
  return null
}

const itemKey = (item: TopicTab) => `${item.id}:${item.label}:${item.count}:${item.shortcut}`

export class Panel extends Component {
  readonly el: HTMLElement
  private buttons = new Map<CameraAction, HTMLButtonElement>()
  private fullscreenPath!: SVGPathElement
  private box: HTMLElement
  private switchEl: HTMLElement
  private tabs: HTMLButtonElement[] = []
  private bodies = new Map<FilterPanel, HTMLElement>()
  private topicsEl: HTMLElement
  private tagsEl: HTMLElement
  private topicButtons: HTMLButtonElement[] = []
  private tagButtons: HTMLButtonElement[] = []
  private topicIds = ''
  private tagIds = ''

  constructor(
    parent: HTMLElement,
    private host: PanelHost,
  ) {
    super()
    const id = `bookcase-${++views}`
    this.el = parent.createDiv({ cls: 'bookcase-controls' })
    const camera = this.el.createDiv({
      cls: 'bookcase-camera-controls',
      attr: { role: 'group', 'aria-label': 'Bookcase controls' },
    })
    for (const action of Object.keys(ICONS) as CameraAction[]) {
      const button = camera.createEl('button', {
        attr: {
          type: 'button',
          'data-camera': action,
          'aria-label': LABELS[action],
          title: LABELS[action],
        },
      })
      const path = icon(button, ICONS[action])
      if (action === 'fullscreen') {
        button.setAttribute('aria-pressed', 'false')
        this.fullscreenPath = path
      }
      this.buttons.set(action, button)
    }

    this.box = this.el.createDiv({ cls: 'bookcase-panel' })
    this.switchEl = this.box.createDiv({
      cls: 'bookcase-panel-tabs',
      attr: { role: 'tablist', 'aria-label': 'Filters' },
    })
    for (const name of PANELS) {
      this.tabs.push(
        this.switchEl.createEl('button', {
          cls: 'bookcase-panel-tab',
          text: name,
          attr: {
            type: 'button',
            role: 'tab',
            id: `${id}-${name}-tab`,
            'aria-controls': `${id}-${name}`,
            'data-panel': name,
          },
        }),
      )
      this.bodies.set(
        name,
        this.box.createDiv({
          cls: 'bookcase-panel-body',
          attr: { role: 'tabpanel', id: `${id}-${name}`, 'aria-labelledby': `${id}-${name}-tab` },
        }),
      )
    }
    this.topicsEl = this.bodies
      .get('topics')!
      .createDiv({
        cls: 'bookcase-topics',
        attr: { role: 'tablist', 'aria-label': 'Topics', 'aria-orientation': 'vertical' },
      })
    this.tagsEl = this.bodies
      .get('tags')!
      .createDiv({
        cls: 'bookcase-tags',
        attr: { role: 'toolbar', 'aria-label': 'Tags', 'aria-orientation': 'vertical' },
      })
  }

  onload(): void {
    this.registerDomEvent(this.el, 'click', evt => {
      const target = evt.target as Element
      const camera = target.closest<HTMLElement>('button[data-camera]')?.dataset.camera
      if (camera) return this.host.camera(camera as CameraAction)
      const panel = target.closest<HTMLElement>('button[data-panel]')?.dataset.panel
      if (panel) return this.host.panel(panel as FilterPanel)
      const topic = target.closest<HTMLElement>('button[data-topic]')?.dataset.topic
      if (topic !== undefined) return this.host.topic(topic)
      const tag = target.closest<HTMLElement>('button[data-tag]')?.dataset.tag
      if (tag !== undefined) this.host.tag(tag)
    })
    // Tabs follow the arrows at once; a letter jumps to the next tab with that shortcut. Tag
    // toggles take the same keys but only move focus; Enter or Space presses them.
    this.registerDomEvent(this.box, 'keydown', evt => {
      if (evt.ctrlKey || evt.metaKey || evt.altKey || evt.isComposing) return
      const target = evt.target as HTMLButtonElement
      let i = this.tabs.indexOf(target)
      if (i >= 0) {
        const next = step(evt.key, i, this.tabs.length)
        if (next === null) return
        evt.preventDefault()
        this.host.panel(PANELS[next])
        this.tabs[next].focus()
        return
      }
      i = this.topicButtons.indexOf(target)
      if (i >= 0) {
        const next =
          step(evt.key, i, this.topicButtons.length) ?? letter(evt.key, i, this.topicButtons)
        if (next === null) return
        evt.preventDefault()
        this.host.topic(this.topicButtons[next].dataset.topic!)
        this.topicButtons[next].focus()
        return
      }
      i = this.tagButtons.indexOf(target)
      if (i < 0) return
      const next = step(evt.key, i, this.tagButtons.length) ?? letter(evt.key, i, this.tagButtons)
      if (next === null) return
      evt.preventDefault()
      this.tagButtons[i].setAttribute('tabindex', '-1')
      this.tagButtons[next].setAttribute('tabindex', '0')
      this.tagButtons[next].focus()
    })
  }

  setCamera(state: { canIn: boolean; canOut: boolean; fullscreen: boolean }): void {
    this.buttons.get('in')!.disabled = !state.canIn
    this.buttons.get('out')!.disabled = !state.canOut
    const full = this.buttons.get('fullscreen')!
    if (full.getAttribute('aria-pressed') === String(state.fullscreen)) return
    const label = state.fullscreen ? 'Exit full screen' : LABELS.fullscreen
    full.setAttribute('aria-pressed', String(state.fullscreen))
    full.setAttribute('aria-label', label)
    full.setAttribute('title', label)
    this.fullscreenPath.setAttribute('d', state.fullscreen ? EXIT_ICON : ICONS.fullscreen)
  }

  /** Tabs and toggles for the filter on show; the buttons are rebuilt only when their set changes. */
  render(tabs: TopicTab[], chips: TopicTab[], filter: ViewFilter): void {
    const has = { topics: tabs.length > 0, tags: chips.length > 0 }
    this.box.toggleClass('is-hidden', !has.topics && !has.tags)
    // With only one kind to list, the column shows it without a switch.
    this.switchEl.toggleClass('is-hidden', !has.topics || !has.tags)
    const shown = has[filter.panel] ? filter.panel : has.topics ? 'topics' : 'tags'
    for (const [i, name] of PANELS.entries()) {
      const tab = this.tabs[i]
      tab.setAttribute('aria-selected', String(name === shown))
      tab.setAttribute('tabindex', name === shown ? '0' : '-1')
      this.bodies.get(name)!.toggleClass('is-hidden', name !== shown)
    }

    // A column is built when it is first shown, so a hidden one costs the first render nothing.
    const topicIds = shown === 'topics' ? tabs.map(itemKey).join('|') : this.topicIds
    if (topicIds !== this.topicIds) {
      this.topicIds = topicIds
      this.topicButtons = this.column(this.topicsEl, tabs, 'topic')
    }
    for (const button of this.topicButtons) {
      const selected = button.dataset.topic === filter.topic
      button.setAttribute('aria-selected', String(selected))
      button.setAttribute('tabindex', selected ? '0' : '-1')
    }

    const tagIds = shown === 'tags' ? chips.map(itemKey).join('|') : this.tagIds
    if (tagIds !== this.tagIds) {
      this.tagIds = tagIds
      this.tagButtons = this.column(this.tagsEl, chips, 'tag')
    }
    for (const [i, button] of this.tagButtons.entries()) {
      button.setAttribute('aria-pressed', String(filter.tags.includes(button.dataset.tag!)))
      if (!this.tagsEl.contains(document.activeElement))
        button.setAttribute('tabindex', i === 0 ? '0' : '-1')
    }
  }

  /**
   * Stage px the column takes from the right edge: its inset and its widest button, an open tab
   * at its full width even while it is still opening.
   */
  reserve(): number {
    if (this.box.hasClass('is-hidden')) return 0
    let width = 0
    for (const button of Array.from(this.el.querySelectorAll<HTMLElement>('.bookcase-tab'))) {
      if (button.offsetParent === null) continue
      const label = button.querySelector<HTMLElement>('.bookcase-tab-label')
      const opening = button.matches('[aria-selected="true"], [aria-pressed="true"]') && label
      width = Math.max(
        width,
        opening ? button.offsetWidth - label.clientWidth + label.scrollWidth : button.offsetWidth,
      )
    }
    return width + (parseFloat(getComputedStyle(this.el).right) || 0)
  }

  private column(el: HTMLElement, items: TopicTab[], kind: 'topic' | 'tag'): HTMLButtonElement[] {
    el.empty()
    return items.map(item => {
      const name = `${item.label} ${item.count}`
      const button = el.createEl('button', {
        cls: `bookcase-tab bookcase-${kind}`,
        attr: {
          type: 'button',
          [`data-${kind}`]: item.id,
          'data-shortcut': item.shortcut[0].toLowerCase(),
          'aria-label': `${item.label}, ${item.count} books`,
          title: item.label,
        },
      })
      if (kind === 'topic') button.setAttribute('role', 'tab')
      button.setCssProps({
        '--bc-tab-short': `${item.shortcut.length}ch`,
        '--bc-tab-label': `${name.length}ch`,
      })
      button.createSpan({
        cls: 'bookcase-tab-shortcut',
        text: item.shortcut,
        attr: { 'aria-hidden': 'true' },
      })
      button.createSpan({ cls: 'bookcase-tab-label', text: name, attr: { 'aria-hidden': 'true' } })
      return button
    })
  }
}
