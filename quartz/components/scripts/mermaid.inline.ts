import type { Mermaid } from 'mermaid'
import { removeAllChildren } from './util'

interface Position {
  x: number
  y: number
}

class DiagramPanZoom {
  private isDragging = false
  private startPan: Position = { x: 0, y: 0 }
  private currentPan: Position = { x: 0, y: 0 }
  private scale = 1
  private readonly MIN_SCALE = 0.5
  private readonly MAX_SCALE = 3

  cleanups: (() => void)[] = []

  constructor(
    private container: HTMLElement,
    private content: HTMLElement,
  ) {
    this.setupEventListeners()
    this.setupNavigationControls()
    this.resetTransform()
  }

  private setupEventListeners() {
    // Mouse drag events
    const mouseDownHandler = this.onMouseDown.bind(this)
    const mouseMoveHandler = this.onMouseMove.bind(this)
    const mouseUpHandler = this.onMouseUp.bind(this)
    const resizeHandler = this.resetTransform.bind(this)
    // Touch drag events
    const touchStartHandler = this.onTouchStart.bind(this)
    const touchMoveHandler = this.onTouchMove.bind(this)
    const touchEndHandler = this.onTouchEnd.bind(this)

    this.container.addEventListener('mousedown', mouseDownHandler)
    document.addEventListener('mousemove', mouseMoveHandler)
    document.addEventListener('mouseup', mouseUpHandler)
    window.addEventListener('resize', resizeHandler)
    this.container.addEventListener('touchstart', touchStartHandler, { passive: false })
    document.addEventListener('touchmove', touchMoveHandler, { passive: false })
    document.addEventListener('touchend', touchEndHandler)

    this.cleanups.push(
      () => this.container.removeEventListener('mousedown', mouseDownHandler),
      () => document.removeEventListener('mousemove', mouseMoveHandler),
      () => document.removeEventListener('mouseup', mouseUpHandler),
      () => window.removeEventListener('resize', resizeHandler),
      () => this.container.removeEventListener('touchstart', touchStartHandler),
      () => document.removeEventListener('touchmove', touchMoveHandler),
      () => document.removeEventListener('touchend', touchEndHandler),
    )
  }

  cleanup() {
    for (const cleanup of this.cleanups) {
      cleanup()
    }
  }

  private setupNavigationControls() {
    const controls = document.createElement('div')
    controls.className = 'mermaid-controls'

    // Zoom controls
    const zoomIn = this.createButton('+', () => this.zoom(0.1))
    const zoomOut = this.createButton('-', () => this.zoom(-0.1))
    const resetBtn = this.createButton('Reset', () => this.resetTransform())

    controls.appendChild(zoomOut)
    controls.appendChild(resetBtn)
    controls.appendChild(zoomIn)

    this.container.appendChild(controls)
  }

  private createButton(text: string, onClick: () => void): HTMLButtonElement {
    const button = document.createElement('button')
    button.textContent = text
    button.className = 'mermaid-control-button'
    button.addEventListener('click', onClick)
    this.cleanups.push(() => button.removeEventListener('click', onClick))
    return button
  }

  private onMouseDown(e: MouseEvent) {
    if (e.button !== 0) return // Only handle left click
    this.isDragging = true
    this.startPan = { x: e.clientX - this.currentPan.x, y: e.clientY - this.currentPan.y }
    this.container.style.cursor = 'grabbing'
  }

  private onMouseMove(e: MouseEvent) {
    if (!this.isDragging) return
    e.preventDefault()

    this.currentPan = { x: e.clientX - this.startPan.x, y: e.clientY - this.startPan.y }

    this.updateTransform()
  }

  private onMouseUp() {
    this.isDragging = false
    this.container.style.cursor = 'grab'
  }

  private onTouchStart(e: TouchEvent) {
    if (e.touches.length !== 1) return
    this.isDragging = true
    const touch = e.touches[0]
    this.startPan = { x: touch.clientX - this.currentPan.x, y: touch.clientY - this.currentPan.y }
  }

  private onTouchMove(e: TouchEvent) {
    if (!this.isDragging || e.touches.length !== 1) return
    e.preventDefault() // Prevent scrolling

    const touch = e.touches[0]
    this.currentPan = { x: touch.clientX - this.startPan.x, y: touch.clientY - this.startPan.y }

    this.updateTransform()
  }

  private onTouchEnd() {
    this.isDragging = false
  }

  private zoom(delta: number) {
    const newScale = Math.min(Math.max(this.scale + delta, this.MIN_SCALE), this.MAX_SCALE)

    // Zoom around center
    const rect = this.content.getBoundingClientRect()
    const centerX = rect.width / 2
    const centerY = rect.height / 2

    const scaleDiff = newScale - this.scale
    this.currentPan.x -= centerX * scaleDiff
    this.currentPan.y -= centerY * scaleDiff

    this.scale = newScale
    this.updateTransform()
  }

  private updateTransform() {
    this.content.style.transform = `translate(${this.currentPan.x}px, ${this.currentPan.y}px) scale(${this.scale})`
  }

  private resetTransform() {
    this.scale = 1
    const svg = this.content.querySelector('svg')!
    const containerRect = this.container.getBoundingClientRect()
    const rect = svg.getBoundingClientRect()
    const width = rect.width / this.scale
    const height = rect.height / this.scale

    this.scale = 1
    this.currentPan = {
      x: (containerRect.width - width) / 2,
      y: (containerRect.height - height) / 2,
    }
    this.updateTransform()
  }
}

const cssVars = [
  '--secondary',
  '--tertiary',
  '--gray',
  '--light',
  '--lightgray',
  '--highlight',
  '--dark',
  '--darkgray',
  '--codeFont',
] as const

let mermaidImport = undefined
let isSetup = false
let textMapping: WeakMap<HTMLElement, string> = new WeakMap()
let panZoomInstances: WeakMap<HTMLElement, DiagramPanZoom> = new WeakMap()
let popupObserver: MutationObserver | null = null
let setupDiagrams: WeakSet<HTMLElement> = new WeakSet()
let observedPopupContainers: WeakSet<Element> = new WeakSet()

function mermaidBlocks(root: Document | HTMLElement = document): HTMLDivElement[] {
  const isSlides = document.documentElement.getAttribute('data-slides') === 'true'
  if (root === document && isSlides) {
    const activeSlide = document.querySelector('.slide.active')
    if (!activeSlide) return []
    return Array.from(activeSlide.querySelectorAll<HTMLDivElement>('pre:has(code.mermaid)'))
  }

  return Array.from(root.querySelectorAll<HTMLDivElement>('pre:has(code.mermaid)'))
}

async function renderMermaidDiagrams(root: Document | HTMLElement = document) {
  const nodes = mermaidBlocks(root)
  if (nodes.length === 0) return

  mermaidImport ||= await import(
    // @ts-ignore
    'https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.12.0/mermaid.esm.min.mjs'
  )
  const mermaid: Mermaid = mermaidImport.default

  // de-init any other diagrams
  for (const node of nodes) {
    const n = node.querySelector('code.mermaid') as HTMLDivElement
    n.removeAttribute('data-processed')
    const oldText = textMapping.get(n)
    if (oldText !== undefined) {
      n.textContent = oldText
    }
  }

  const computedStyleMap = cssVars.reduce(
    (acc, key) => {
      acc[key] = window.getComputedStyle(document.documentElement).getPropertyValue(key)
      return acc
    },
    {} as Record<(typeof cssVars)[number], string>,
  )

  const darkMode = document.documentElement.getAttribute('saved-theme') === 'dark'
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: darkMode ? 'dark' : 'base',
    themeVariables: {
      fontFamily: computedStyleMap['--codeFont'],
      primaryColor: computedStyleMap['--light'],
      primaryTextColor: computedStyleMap['--darkgray'],
      primaryBorderColor: computedStyleMap['--tertiary'],
      lineColor: computedStyleMap['--darkgray'],
      secondaryColor: computedStyleMap['--secondary'],
      tertiaryColor: computedStyleMap['--tertiary'],
      clusterBkg: computedStyleMap['--light'],
      edgeLabelBackground: computedStyleMap['--highlight'],
    },
  })

  await mermaid.run({
    nodes: [...nodes].map(n => n.querySelector('code.mermaid') as HTMLDivElement),
  })
}

async function setupMermaid(root: Document | HTMLElement = document) {
  const stackedContainer = document.getElementById('stacked-notes-container')
  if (root === document && stackedContainer?.classList.contains('active')) return

  const nodes = mermaidBlocks(root)
  if (nodes.length === 0) return

  for (const node of nodes) {
    const n = node.querySelector('code.mermaid') as HTMLDivElement
    if (!textMapping.has(n)) {
      textMapping.set(n, n.textContent ?? '')
    }
  }

  await renderMermaidDiagrams(root)

  if (!isSetup) {
    const rerenderMermaid = () => {
      void renderMermaidDiagrams()
    }
    document.addEventListener('themechange', rerenderMermaid)
    window.addCleanup(() => {
      document.removeEventListener('themechange', rerenderMermaid)
      popupObserver?.disconnect()
      popupObserver = null
      observedPopupContainers = new WeakSet()
      isSetup = false
    })
    isSetup = true
  }

  for (let i = 0; i < nodes.length; i++) {
    const pre = nodes[i]
    if (setupDiagrams.has(pre)) continue

    const codeBlock = pre.querySelector('code.mermaid') as HTMLDivElement
    const clipboardBtn = pre.querySelector('.clipboard-button') as HTMLElement | null
    const expandBtn = pre.querySelector('.expand-button') as HTMLElement | null

    if (!(clipboardBtn instanceof Element) || !(expandBtn instanceof HTMLElement)) {
      continue
    }

    let clipboardWidth = 0
    try {
      const clipboardStyle = window.getComputedStyle(clipboardBtn)
      clipboardWidth =
        (clipboardBtn as HTMLElement).offsetWidth +
        parseFloat(clipboardStyle.marginLeft || '0') +
        parseFloat(clipboardStyle.marginRight || '0')
    } catch {
      clipboardWidth = clipboardBtn.offsetWidth || 0
    }

    expandBtn.style.right = `calc(${clipboardWidth}px + 0.3rem)`

    const popupContainer = pre.querySelector('#mermaid-container') as HTMLElement | null
    if (!popupContainer) {
      continue
    }

    function showMermaid() {
      const container = popupContainer!.querySelector('#mermaid-space') as HTMLElement
      const content = popupContainer!.querySelector('.mermaid-content') as HTMLElement
      if (!content) return
      removeAllChildren(content)

      const mermaidContent = codeBlock.querySelector('svg')!.cloneNode(true) as SVGElement
      content.appendChild(mermaidContent)

      popupContainer?.classList.add('active')
      container.style.cursor = 'grab'

      const panZoom = new DiagramPanZoom(container, content)
      panZoomInstances.set(popupContainer!, panZoom)
    }

    function hideMermaid() {
      popupContainer?.classList.remove('active')
      const panZoom = panZoomInstances.get(popupContainer!)
      panZoom?.cleanup()
      panZoomInstances.delete(popupContainer!)
    }

    const onEscape = (event: KeyboardEvent) => {
      if (!event.key.startsWith('Esc')) return
      event.preventDefault()
      hideMermaid()
    }
    const onOutsideClick = (event: MouseEvent) => {
      if (event.target !== popupContainer) return
      event.preventDefault()
      event.stopPropagation()
      hideMermaid()
    }

    expandBtn.addEventListener('click', showMermaid)
    popupContainer.addEventListener('click', onOutsideClick)
    document.addEventListener('keydown', onEscape)

    window.addCleanup(() => {
      const panZoom = panZoomInstances.get(popupContainer!)
      panZoom?.cleanup()
      expandBtn.removeEventListener('click', showMermaid)
      popupContainer.removeEventListener('click', onOutsideClick)
      document.removeEventListener('keydown', onEscape)
    })

    setupDiagrams.add(pre)
  }

  if (!popupObserver) {
    popupObserver = new MutationObserver(mutations => {
      for (const mutation of mutations) {
        if (
          mutation.type === 'attributes' &&
          mutation.attributeName === 'class' &&
          mutation.target instanceof HTMLElement &&
          mutation.target.id === 'mermaid-container'
        ) {
          if (!mutation.target.classList.contains('active')) {
            const panZoom = panZoomInstances.get(mutation.target)
            panZoom?.cleanup()
            panZoomInstances.delete(mutation.target)
          }
        }
      }
    })
  }

  root.querySelectorAll('#mermaid-container').forEach(container => {
    if (observedPopupContainers.has(container)) return
    popupObserver!.observe(container, { attributes: true, attributeFilter: ['class'] })
    observedPopupContainers.add(container)
  })
}
document.addEventListener('nav', () => {
  void setupMermaid()
})
document.addEventListener('contentdecrypted', event => {
  void setupMermaid(event.detail.content)
})
document.addEventListener('slidechange', () => {
  void setupMermaid()
})
