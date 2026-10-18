// Mouse and pen drag on the stage pans the camera, as in the garden. A touch scrolls the track
// instead, since the scroll is what moves along the books. A press that moves less than
// THRESHOLD px stays a click; after a drag the click is swallowed, so no book opens.
import { Component } from 'obsidian'
import type { Point } from './scene-model'

const THRESHOLD = 5

export interface DragHost {
  /** Current pan, read when a drag starts. */
  pan(): Point
  move(pan: Point): void
  start(): void
}

interface Gesture {
  id: number
  x: number
  y: number
  pan: Point
  moved: boolean
}

export class Drag extends Component {
  private gesture: Gesture | null = null
  private swallow = false
  private timer: number | null = null

  constructor(
    private el: HTMLElement,
    /** Presses inside this selector are not drags (the controls). */
    private ignore: string,
    private host: DragHost,
  ) {
    super()
  }

  get dragging(): boolean {
    return this.gesture?.moved ?? false
  }

  onload(): void {
    this.registerDomEvent(this.el, 'pointerdown', evt => {
      this.swallow = false
      if (!evt.isPrimary || evt.button !== 0 || evt.pointerType === 'touch') return
      if ((evt.target as Element).closest(this.ignore)) return
      this.gesture = {
        id: evt.pointerId,
        x: evt.clientX,
        y: evt.clientY,
        pan: this.host.pan(),
        moved: false,
      }
    })
    this.registerDomEvent(this.el, 'pointermove', evt => {
      const g = this.gesture
      if (!g || evt.pointerId !== g.id) return
      const dx = evt.clientX - g.x
      const dy = evt.clientY - g.y
      if (!g.moved) {
        if (Math.hypot(dx, dy) < THRESHOLD) return
        // Captured only once it is a drag, so a plain press still clicks the book under it.
        g.moved = true
        this.el.setPointerCapture(g.id)
        this.el.dataset.dragging = ''
        this.host.start()
      }
      this.host.move([g.pan[0] + dx, g.pan[1] + dy])
    })
    this.registerDomEvent(this.el, 'pointerup', evt => {
      const g = this.gesture
      if (!g || evt.pointerId !== g.id) return
      if (g.moved) {
        this.swallow = true
        // The click, if any, comes in the same task; a press that ends elsewhere has none.
        if (this.timer !== null) window.clearTimeout(this.timer)
        this.timer = window.setTimeout(() => {
          this.timer = null
          this.swallow = false
        }, 0)
      }
      this.end()
    })
    this.registerDomEvent(this.el, 'pointercancel', () => this.end())
    this.registerDomEvent(this.el, 'lostpointercapture', () => this.end())
    this.registerDomEvent(
      this.el,
      'click',
      evt => {
        if (!this.swallow) return
        this.swallow = false
        evt.preventDefault()
        evt.stopPropagation()
      },
      { capture: true },
    )
  }

  onunload(): void {
    if (this.timer !== null) window.clearTimeout(this.timer)
    this.timer = null
    this.end()
  }

  private end(): void {
    const g = this.gesture
    if (!g) return
    this.gesture = null
    delete this.el.dataset.dragging
    if (this.el.hasPointerCapture(g.id)) this.el.releasePointerCapture(g.id)
  }
}
