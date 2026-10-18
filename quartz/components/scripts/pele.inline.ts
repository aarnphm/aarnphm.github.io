import { mount, mountAsync, type Mounted } from 'pele/lazy'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

function diagramFigures(root: Document | HTMLElement): HTMLElement[] {
  if (root instanceof HTMLElement && root.matches('figure.pele-figure')) return [root]
  if (root === document && document.documentElement.dataset.slides === 'true') {
    return Array.from(document.querySelectorAll('.slide.active figure.pele-figure'))
  }
  return Array.from(root.querySelectorAll('figure.pele-figure'))
}

document.addEventListener('nav', () => {
  const diagrams = new WeakSet<HTMLElement>()
  const navSignal = rootNavSignal(document)

  async function setupDiagram(figure: HTMLElement) {
    if (diagrams.has(figure)) return
    const drawing = figure.querySelector('.pele-diagram')
    const expand = figure.querySelector('button.pele-trigger')
    const copy = figure.querySelector('button.pele-copy')
    const dialog = figure.querySelector('dialog.pele-dialog')
    const content = dialog?.querySelector('.pele-dialog-content')
    const close = dialog?.querySelector('button.pele-close')
    const controls = dialog?.querySelector('.pele-controls')
    const fold = dialog?.querySelector<HTMLButtonElement>('button.pele-fold')
    const controlBody = dialog?.querySelector<HTMLElement>('.pele-controls-body')
    const zoomIn = dialog?.querySelector<HTMLButtonElement>('button.pele-zoom-in')
    const zoomOut = dialog?.querySelector<HTMLButtonElement>('button.pele-zoom-out')
    const reset = dialog?.querySelector<HTMLButtonElement>('button.pele-reset')
    if (
      !(drawing instanceof HTMLElement) ||
      !(expand instanceof HTMLButtonElement) ||
      !(copy instanceof HTMLButtonElement) ||
      !(dialog instanceof HTMLDialogElement) ||
      !(content instanceof HTMLElement) ||
      !(close instanceof HTMLButtonElement) ||
      !(controls instanceof HTMLElement) ||
      !fold ||
      !controlBody ||
      !zoomIn ||
      !zoomOut ||
      !reset
    )
      return

    const signal = rootNavSignal(figure)
    const source = drawing.dataset.clipboard ?? drawing.textContent ?? ''
    let diagram: Mounted | undefined
    let expanded: Mounted | undefined
    let copiedTimer: ReturnType<typeof setTimeout> | undefined
    let zoomObserver: MutationObserver | undefined
    diagrams.add(figure)

    copy.addEventListener(
      'click',
      () => {
        void navigator.clipboard.writeText(source).then(
          () => {
            if (signal.aborted) return
            copy.setAttribute('aria-label', 'source copied')
            copy.title = 'source copied'
            copy.classList.add('check')
            clearTimeout(copiedTimer)
            copiedTimer = setTimeout(() => {
              copy.classList.remove('check')
              copy.setAttribute('aria-label', 'copy source')
              copy.title = 'copy source'
            }, 2000)
          },
          () => {
            if (!signal.aborted) copy.setAttribute('aria-label', 'copy failed, try again')
          },
        )
      },
      { signal },
    )

    function clearExpanded() {
      zoomObserver?.disconnect()
      zoomObserver = undefined
      expanded?.destroy()
      expanded = undefined
      content?.replaceChildren()
    }
    function closeDiagram() {
      if (dialog instanceof HTMLDialogElement && dialog.open) dialog.close()
      clearExpanded()
    }
    close.addEventListener('click', closeDiagram, { signal })
    dialog.addEventListener(
      'close',
      () => {
        if (!dialog.open) clearExpanded()
      },
      { signal },
    )
    const disposeEscape = registerEscapeHandler(dialog, closeDiagram)

    const syncZoomControls = () => {
      const zoom = expanded?.zoom
      zoomIn.disabled = !zoom || zoom.scale >= Math.max(zoom.fit, 3)
      zoomOut.disabled = !zoom || zoom.scale <= zoom.fit
      reset.disabled = !zoom || zoom.scale <= zoom.fit
    }
    const setControlsExpanded = (open: boolean) => {
      fold.setAttribute('aria-expanded', String(open))
      const label = open ? 'collapse diagram controls' : 'expand diagram controls'
      fold.setAttribute('aria-label', label)
      fold.title = label
      controlBody.hidden = !open
    }
    fold.addEventListener('click', () => setControlsExpanded(controlBody.hidden === true), {
      signal,
    })
    zoomIn.addEventListener('click', () => expanded?.zoom?.zoomBy(1.25), { signal })
    zoomOut.addEventListener('click', () => expanded?.zoom?.zoomBy(1 / 1.25), { signal })
    reset.addEventListener('click', () => expanded?.zoom?.reset(), { signal })
    expand.addEventListener(
      'click',
      () => {
        if (dialog.open) return
        setControlsExpanded(true)
        dialog.showModal()
        expanded = mount(content, source, { zoom: { always: true, controls: false } })
        zoomObserver = new MutationObserver(syncZoomControls)
        zoomObserver.observe(content, {
          attributes: true,
          attributeFilter: ['viewBox'],
          childList: true,
          subtree: true,
        })
        syncZoomControls()
      },
      { signal },
    )

    signal.addEventListener(
      'abort',
      () => {
        closeDiagram()
        diagram?.destroy()
        diagrams.delete(figure)
        clearTimeout(copiedTimer)
        disposeEscape()
      },
      { once: true },
    )

    try {
      diagram = await mountAsync(drawing, source, { zoom: false })
    } catch (error) {
      if (signal.aborted || !figure.isConnected) return
      const message = document.createElement('p')
      message.className = 'pele-load-error'
      message.setAttribute('role', 'status')
      message.textContent = 'diagram unavailable. reload to try again, or copy the source.'
      drawing.replaceChildren(document.createTextNode(source), message)
      expand.disabled = false
      console.error('Unable to load Pele diagram', error)
      return
    }
    if (signal.aborted || !figure.isConnected) {
      diagram.destroy()
      return
    }
    drawing.dataset.renderer = 'pele'
    expand.disabled = false
  }

  function setup(root: Document | HTMLElement = document) {
    if (
      root === document &&
      document.getElementById('stacked-notes-container')?.classList.contains('active')
    )
      return
    for (const figure of diagramFigures(root)) {
      // Canvas cards announce their own content and wait until the diagram is visible.
      if (root === document && figure.closest('.node-file-content') && !diagrams.has(figure))
        continue
      void setupDiagram(figure)
    }
  }

  document.addEventListener(
    'contentdecrypted',
    event => {
      const root = event.detail.content
      const card = root.closest<HTMLElement>('.node-file-content')
      if (!card) {
        setup(root)
        return
      }
      const signal = rootNavSignal(root)
      const observer = new IntersectionObserver(
        entries => {
          for (const entry of entries) {
            if (!entry.isIntersecting || !(entry.target instanceof HTMLElement)) continue
            observer.unobserve(entry.target)
            if (!signal.aborted && entry.target.isConnected) void setupDiagram(entry.target)
          }
        },
        { root: card, rootMargin: '160px' },
      )
      for (const figure of diagramFigures(root)) observer.observe(figure)
      signal.addEventListener('abort', () => observer.disconnect(), { once: true })
    },
    { signal: navSignal },
  )
  document.addEventListener('slidechange', () => setup(), { signal: navSignal })
  setup()
})
