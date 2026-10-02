import type { RoughAnnotation } from 'rough-notation/lib/model'
import { Component, MarkdownRenderChild } from 'obsidian'
import { annotate } from 'rough-notation'
import type MarkedPlugin from '../main'
import type { Intensity } from './parser'

const PALETTE = {
  h1: ['--rose', '#d14d41'],
  h2: ['--love', '#da702c'],
  h3: ['--lime', '#879a39'],
  h4: ['--gold', '#d0a215'],
  h5: ['--pine', '#3aa99f'],
  h6: ['--foam', '#4385be'],
  h7: ['--iris', '#8b7ec8'],
}

export class MarkerAnnotations extends Component {
  private markers = new Set<MarkerRenderChild>()
  private containers = new Map<Element, number>()
  private observer = new ResizeObserver(() => this.schedule())
  private visibility = new IntersectionObserver(() => this.schedule())
  private frame = 0
  private active = false

  constructor(readonly plugin: MarkedPlugin) {
    super()
  }

  onload(): void {
    this.active = true
    this.registerDomEvent(window, 'resize', () => this.schedule())
    this.register(() => {
      this.active = false
      cancelAnimationFrame(this.frame)
      this.observer.disconnect()
      this.visibility.disconnect()
      for (const marker of this.markers) marker.removeAnnotation()
      this.markers.clear()
      this.containers.clear()
    })
    void document.fonts.ready.then(() => this.schedule())
  }

  add(marker: MarkerRenderChild): void {
    this.markers.add(marker)
    this.visibility.observe(marker.containerEl)
    this.schedule()
  }

  remove(marker: MarkerRenderChild): void {
    this.markers.delete(marker)
    this.visibility.unobserve(marker.containerEl)
    if (marker.observedContainer) {
      const count = (this.containers.get(marker.observedContainer) ?? 1) - 1
      if (count > 0) this.containers.set(marker.observedContainer, count)
      else {
        this.observer.unobserve(marker.observedContainer)
        this.containers.delete(marker.observedContainer)
      }
    }
  }

  schedule(): void {
    if (!this.active || this.frame) return
    this.frame = requestAnimationFrame(() => {
      this.frame = 0
      for (const marker of this.markers) {
        if (!marker.observedContainer && marker.containerEl.isConnected) {
          const container =
            marker.containerEl.closest(
              '.markdown-preview-sizer, .cm-content, .markdown-embed-content',
            ) ?? marker.containerEl.parentElement
          if (container) {
            marker.observedContainer = container
            this.containers.set(container, (this.containers.get(container) ?? 0) + 1)
            this.observer.observe(container)
          }
        }
        marker.draw()
      }
    })
  }
}

export class MarkerRenderChild extends MarkdownRenderChild {
  readonly target: HTMLElement
  observedContainer?: Element
  ready = false
  private annotation?: RoughAnnotation
  private signature = ''

  constructor(
    element: HTMLElement,
    readonly intensity: Intensity,
    private manager: MarkerAnnotations,
    private restore?: () => void,
  ) {
    super(element)
    element.className = 'garden-marked'
    element.dataset.intensity = intensity
    this.target = element.createSpan({ cls: 'garden-marked-text' })
  }

  onload(): void {
    this.manager.add(this)
  }

  onunload(): void {
    this.removeAnnotation()
    this.manager.remove(this)
    this.restore?.()
  }

  removeAnnotation(): void {
    this.annotation?.remove()
    this.annotation = undefined
    this.signature = ''
  }

  draw(): void {
    if (!this.ready || !this.containerEl.isConnected || !this.target.getBoundingClientRect().width)
      return
    const settings = this.manager.plugin.settings
    const level = settings.levels[this.intensity]
    const type = level.type === 'default' ? settings.type : level.type
    const style = getComputedStyle(this.target)
    const [variable, fallback] = PALETTE[this.intensity]
    const color = level.color || style.getPropertyValue(variable).trim() || fallback
    const signature = `${type}:${color}`
    if (signature !== this.signature) {
      this.removeAnnotation()
      this.containerEl.dataset.annotationType = type
      this.annotation = annotate(this.target, {
        type,
        color,
        animate: false,
        iterations: 2,
        strokeWidth: 1.5,
        padding: 2,
        multiline: true,
        brackets: ['left', 'right'],
      })
      const svg = this.containerEl.querySelector('.rough-annotation')
      svg?.setAttribute('aria-hidden', 'true')
      svg?.setAttribute('data-garden-marked', '')
      this.signature = signature
    }
    this.annotation?.show()
  }
}
