type Zoom = 'fit' | number

export function mountPreview(dialog: HTMLDialogElement, signal: AbortSignal) {
  const viewport = dialog.querySelector<HTMLElement>('.press-kit-viewport')
  const frame = dialog.querySelector<HTMLElement>('.press-kit-frame')
  const canvas = dialog.querySelector<HTMLCanvasElement>('.press-kit-canvas')
  const toolbar = dialog.querySelector<HTMLElement>('.press-kit-preview-zoom')
  const percent = toolbar?.querySelector<HTMLOutputElement>('[data-preview-percent]')
  if (!viewport || !frame || !canvas || !toolbar || !percent) return

  const buttons = toolbar.querySelectorAll<HTMLButtonElement>('[data-preview-zoom]')
  let zoom: Zoom = 'fit'
  let scale = 1
  let fit = 1
  let imageWidth = 0
  let imageHeight = 0
  let pending = 0

  const layout = () => {
    pending = 0
    if (!dialog.open || canvas.hidden || !viewport.clientWidth || !viewport.clientHeight) return
    if (canvas.width !== imageWidth || canvas.height !== imageHeight) {
      imageWidth = canvas.width
      imageHeight = canvas.height
      zoom = 'fit'
    }
    const previous = canvas.getBoundingClientRect()
    const view = viewport.getBoundingClientRect()
    const center = {
      x: (view.left + viewport.clientWidth / 2 - previous.left) / previous.width,
      y: (view.top + viewport.clientHeight / 2 - previous.top) / previous.height,
    }
    const padding = getComputedStyle(frame)
    fit = Math.min(
      1,
      (viewport.clientWidth - parseFloat(padding.paddingLeft) - parseFloat(padding.paddingRight)) /
        imageWidth,
      (viewport.clientHeight - parseFloat(padding.paddingTop) - parseFloat(padding.paddingBottom)) /
        imageHeight,
    )
    scale = zoom === 'fit' ? fit : Math.max(Math.min(fit, 0.1), Math.min(2, zoom))
    canvas.style.width = `${imageWidth * scale}px`
    canvas.style.height = `${imageHeight * scale}px`
    if (zoom === 'fit') {
      viewport.scrollLeft = 0
      viewport.scrollTop = 0
    } else {
      const next = canvas.getBoundingClientRect()
      viewport.scrollLeft +=
        next.left + center.x * next.width - view.left - viewport.clientWidth / 2
      viewport.scrollTop += next.top + center.y * next.height - view.top - viewport.clientHeight / 2
    }
    percent.value = `${Math.round(scale * 100)}%`
    for (const button of buttons) {
      const action = button.dataset.previewZoom
      button.disabled =
        (action === 'out' && scale <= Math.min(fit, 0.1) + 0.0001) ||
        (action === 'in' && scale >= 2)
      if (action === 'fit' || action === 'actual')
        button.setAttribute('aria-pressed', String(action === 'fit' ? zoom === 'fit' : zoom === 1))
    }
  }
  const refresh = () => {
    if (!pending) pending = requestAnimationFrame(layout)
  }
  const change = (action: string | undefined) => {
    if (action === 'fit') zoom = 'fit'
    else if (action === 'actual') zoom = 1
    else if (action === 'in') zoom = Math.min(2, (zoom === 'fit' ? scale : zoom) * 1.25)
    else if (action === 'out')
      zoom = Math.max(Math.min(fit, 0.1), (zoom === 'fit' ? scale : zoom) / 1.25)
    else return
    refresh()
  }
  toolbar.addEventListener(
    'click',
    event => {
      if (event.target instanceof Element)
        change(event.target.closest<HTMLButtonElement>('[data-preview-zoom]')?.dataset.previewZoom)
    },
    { signal },
  )
  viewport.addEventListener(
    'keydown',
    event => {
      if (event.target !== viewport || event.ctrlKey || event.metaKey || event.altKey) return
      const action =
        event.key === '+' || event.key === '='
          ? 'in'
          : event.key === '-'
            ? 'out'
            : event.key === '0'
              ? 'fit'
              : null
      if (!action) return
      event.preventDefault()
      event.stopPropagation()
      change(action)
    },
    { signal },
  )

  const resized = new ResizeObserver(refresh)
  resized.observe(viewport)
  const changed = new MutationObserver(records => {
    if (records.some(record => record.target === dialog && record.attributeName === 'open'))
      zoom = 'fit'
    refresh()
  })
  changed.observe(canvas, { attributes: true, attributeFilter: ['width', 'height', 'hidden'] })
  changed.observe(frame, { attributes: true, attributeFilter: ['data-editor-rulers'] })
  changed.observe(dialog, { attributes: true, attributeFilter: ['open', 'data-tab'] })
  signal.addEventListener(
    'abort',
    () => {
      cancelAnimationFrame(pending)
      resized.disconnect()
      changed.disconnect()
      canvas.style.removeProperty('width')
      canvas.style.removeProperty('height')
    },
    { once: true },
  )
  refresh()
}
