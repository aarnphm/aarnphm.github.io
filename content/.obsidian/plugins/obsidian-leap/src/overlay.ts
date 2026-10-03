export interface OverlayTarget {
  rect: DOMRect
  label?: string
  active?: boolean
}

export class LeapOverlay {
  private element?: HTMLElement

  constructor(private document: Document) {}

  show(targets: OverlayTarget[]): void {
    this.clear()
    if (!targets.length) return
    const overlay = this.document.createElement('div')
    overlay.className = 'garden-leap-overlay garden-plugin-ui'
    overlay.setAttribute('aria-hidden', 'true')
    const labelPositions = new Map<string, number>()
    for (const target of targets) {
      const marker = this.document.createElement('span')
      marker.className = 'garden-leap-target'
      marker.classList.toggle('is-active', target.active === true)
      marker.style.left = `${target.rect.left}px`
      marker.style.top = `${target.rect.top}px`
      marker.style.width = `${Math.max(target.rect.width, 2)}px`
      marker.style.height = `${target.rect.height}px`
      if (target.label) {
        const label = this.document.createElement('span')
        label.className = 'garden-leap-label'
        label.textContent = target.label
        const position = `${Math.round(target.rect.left)}:${Math.round(target.rect.top)}`
        const offset = labelPositions.get(position) ?? 0
        label.style.transform = `translateX(${offset * 1.75}ch)`
        labelPositions.set(position, offset + 1)
        marker.append(label)
      }
      overlay.append(marker)
    }
    this.document.body.append(overlay)
    this.element = overlay
  }

  clear(): void {
    this.element?.remove()
    this.element = undefined
  }
}
