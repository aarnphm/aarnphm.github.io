// Camera buttons and, under them, a right-aligned column of topic tabs, as the garden's: a
// collapsed tab shows its shortcut, the chosen one opens leftward to "name count". Note text
// reaches the buttons as text content only.
import { Component } from 'obsidian'
import type { TopicTab } from './topics'

export type CameraAction = 'in' | 'out' | 'fit' | 'fullscreen'

export interface PanelHost {
  camera(action: CameraAction): void
  topic(id: string): void
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

function icon(parent: HTMLElement, d: string): SVGPathElement {
  const svg = parent.createSvg('svg', {
    attr: { viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
  })
  return svg.createSvg('path', { attr: { d } })
}

/** Arrow keys, Home and End along the column: the index they lead to from `i`, or null. */
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
  private topicsEl: HTMLElement
  private topicButtons: HTMLButtonElement[] = []
  private topicIds = ''

  constructor(
    parent: HTMLElement,
    private host: PanelHost,
  ) {
    super()
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
    this.topicsEl = this.box.createDiv({
      cls: 'bookcase-topics',
      attr: { role: 'tablist', 'aria-label': 'Topics', 'aria-orientation': 'vertical' },
    })
  }

  onload(): void {
    this.registerDomEvent(this.el, 'click', evt => {
      const target = evt.target as Element
      const camera = target.closest<HTMLElement>('button[data-camera]')?.dataset.camera
      if (camera) return this.host.camera(camera as CameraAction)
      const topic = target.closest<HTMLElement>('button[data-topic]')?.dataset.topic
      if (topic !== undefined) this.host.topic(topic)
    })
    // Tabs follow the arrows at once; a letter jumps to the next tab with that shortcut.
    this.registerDomEvent(this.box, 'keydown', evt => {
      if (evt.ctrlKey || evt.metaKey || evt.altKey || evt.isComposing) return
      const i = this.topicButtons.indexOf(evt.target as HTMLButtonElement)
      if (i < 0) return
      const next =
        step(evt.key, i, this.topicButtons.length) ?? letter(evt.key, i, this.topicButtons)
      if (next === null) return
      evt.preventDefault()
      this.host.topic(this.topicButtons[next].dataset.topic!)
      this.topicButtons[next].focus()
    })
  }

  /** Selects and focuses the next topic tab matching `key` from anywhere in the bookcase. */
  stepByLetter(key: string): boolean {
    if (this.box.hasClass('is-hidden') || this.topicButtons.length === 0) return false
    const active = this.topicButtons.findIndex(b => b.getAttribute('aria-selected') === 'true')
    const next = letter(key, Math.max(0, active), this.topicButtons)
    if (next === null) return false
    this.host.topic(this.topicButtons[next].dataset.topic!)
    this.topicButtons[next].focus()
    return true
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

  /** Tabs for the topic on show; the buttons are rebuilt only when their set changes. */
  render(tabs: TopicTab[], topic: string): void {
    this.box.toggleClass('is-hidden', tabs.length === 0)
    const topicIds = tabs.map(itemKey).join('|')
    if (topicIds !== this.topicIds) {
      this.topicIds = topicIds
      this.topicButtons = this.column(tabs)
    }
    for (const button of this.topicButtons) {
      const selected = button.dataset.topic === topic
      button.setAttribute('aria-selected', String(selected))
      button.setAttribute('tabindex', selected ? '0' : '-1')
    }
  }

  /**
   * Stage px the column always takes from the right edge: its inset and the camera buttons.
   * Unlike `reserve` it does not follow an opening tab, so the scale that fits beside it holds
   * still.
   */
  inset(): number {
    const camera = this.el.firstElementChild
    const width = camera instanceof HTMLElement ? camera.offsetWidth : 0
    return width + (parseFloat(getComputedStyle(this.el).right) || 0)
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
      const opening = button.matches('[aria-selected="true"]') && label
      width = Math.max(
        width,
        opening ? button.offsetWidth - label.clientWidth + label.scrollWidth : button.offsetWidth,
      )
    }
    return width + (parseFloat(getComputedStyle(this.el).right) || 0)
  }

  private column(items: TopicTab[]): HTMLButtonElement[] {
    this.topicsEl.empty()
    return items.map(item => {
      const name = `${item.label} ${item.count}`
      const button = this.topicsEl.createEl('button', {
        cls: 'bookcase-tab bookcase-topic',
        attr: {
          type: 'button',
          role: 'tab',
          'data-topic': item.id,
          'data-shortcut': item.shortcut[0].toLowerCase(),
          'aria-label': `${item.label}, ${item.count} books`,
          title: item.label,
        },
      })
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
