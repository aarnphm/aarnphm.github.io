import type { PdfControls, PdfFitMode, PdfOptions } from './viewer-model'

const PDF_ICON_PREV =
  'M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z'
const PDF_ICON_NEXT =
  'M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z'
const PDF_ICON_MINUS =
  'M3.5 7C3.22386 7 3 7.22386 3 7.5C3 7.77614 3.22386 8 3.5 8H11.5C11.7761 8 12 7.77614 12 7.5C12 7.22386 11.7761 7 11.5 7H3.5Z'
const PDF_ICON_PLUS =
  'M8 2.75C8 2.47386 7.77614 2.25 7.5 2.25C7.22386 2.25 7 2.47386 7 2.75V7H2.75C2.47386 7 2.25 7.22386 2.25 7.5C2.25 7.77614 2.47386 8 2.75 8H7V12.25C7 12.5261 7.22386 12.75 7.5 12.75C7.77614 12.75 8 12.5261 8 12.25V8H12.25C12.5261 8 12.75 7.77614 12.75 7.5C12.75 7.22386 12.5261 7 12.25 7H8V2.75Z'
const PDF_ICON_DOWNLOAD =
  'M7.50005 1.04999C7.74858 1.04999 7.95005 1.25146 7.95005 1.49999V8.41359L10.1819 6.18179C10.3576 6.00605 10.6425 6.00605 10.8182 6.18179C10.994 6.35753 10.994 6.64245 10.8182 6.81819L7.81825 9.81819C7.64251 9.99392 7.35759 9.99392 7.18185 9.81819L4.18185 6.81819C4.00611 6.64245 4.00611 6.35753 4.18185 6.18179C4.35759 6.00605 4.64251 6.00605 4.81825 6.18179L7.05005 8.41359V1.49999C7.05005 1.25146 7.25152 1.04999 7.50005 1.04999ZM2.5 10C2.77614 10 3 10.2239 3 10.5V12C3 12.5539 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2239 12.2239 10 12.5 10C12.7761 10 13 10.2239 13 10.5V12C13 13.1041 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2239 2.22386 10 2.5 10Z'
const PDF_ICON_SUBMIT =
  'M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5'

interface PdfCommentComposerOptions {
  signal: AbortSignal
  className: string
  placeholder: string
  submitLabel: string
  onSubmit(content: string): void | Promise<void>
  onCancel(): void
}

interface PdfCommentComposer {
  element: HTMLDivElement
  focus(): void
}

function createPdfCommentSubmitIcon(): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('fill', 'none')
  const pathElement = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  pathElement.setAttribute('d', PDF_ICON_SUBMIT)
  pathElement.setAttribute('fill', 'currentColor')
  pathElement.setAttribute('fill-rule', 'evenodd')
  pathElement.setAttribute('clip-rule', 'evenodd')
  svg.append(pathElement)
  return svg
}

function resizePdfCommentInput(input: HTMLTextAreaElement) {
  input.style.height = 'auto'
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`
}

export function createPdfCommentComposer(options: PdfCommentComposerOptions): PdfCommentComposer {
  const element = document.createElement('div')
  element.className = `${options.className} pdf-comment-editor-composer`

  const shell = document.createElement('div')
  shell.className = 'pdf-comment-editor-shell pdf-comment-empty'

  const inputContainer = document.createElement('div')
  inputContainer.className = 'pdf-comment-editor'
  inputContainer.setAttribute('role', 'textbox')
  inputContainer.setAttribute('aria-placeholder', options.placeholder)

  const input = document.createElement('textarea')
  input.className = 'pdf-comment-input'
  input.placeholder = options.placeholder
  input.rows = 1
  input.spellcheck = true
  input.setAttribute('aria-label', options.placeholder)

  const placeholder = document.createElement('div')
  placeholder.className = 'pdf-comment-placeholder'
  placeholder.setAttribute('aria-hidden', 'true')
  const placeholderText = document.createElement('span')
  placeholderText.className = 'pdf-comment-placeholder-text'
  placeholderText.textContent = options.placeholder
  placeholder.appendChild(placeholderText)

  const submit = document.createElement('button')
  submit.type = 'button'
  submit.className = 'pdf-comment-submit'
  submit.ariaLabel = options.submitLabel
  submit.disabled = true
  submit.setAttribute('aria-disabled', 'true')
  const icon = document.createElement('span')
  icon.className = 'pdf-comment-submit-icon'
  icon.appendChild(createPdfCommentSubmitIcon())
  submit.appendChild(icon)

  const sync = () => {
    const empty = input.value.trim().length === 0
    shell.classList.toggle('pdf-comment-empty', empty)
    submit.disabled = empty
    submit.setAttribute('aria-disabled', String(empty))
    resizePdfCommentInput(input)
  }

  let submitting = false
  const save = () => {
    const content = input.value.trim()
    if (submitting || !content) return
    submitting = true
    submit.disabled = true
    submit.setAttribute('aria-disabled', 'true')
    Promise.resolve(options.onSubmit(content))
      .catch(error => {
        console.error('failed to submit pdf comment:', error)
      })
      .finally(() => {
        submitting = false
        if (element.isConnected) sync()
      })
  }

  element.addEventListener('click', event => event.stopPropagation(), { signal: options.signal })
  input.addEventListener('input', sync, { signal: options.signal })
  input.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        options.onCancel()
        return
      }
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        event.stopPropagation()
        save()
        return
      }
      event.stopPropagation()
    },
    { signal: options.signal },
  )
  submit.addEventListener(
    'click',
    event => {
      event.stopPropagation()
      save()
    },
    { signal: options.signal },
  )

  inputContainer.append(input, placeholder)
  shell.append(inputContainer, submit)
  element.appendChild(shell)
  sync()

  return {
    element,
    focus() {
      input.focus()
    },
  }
}

function createButton(className: string, icon: SVGSVGElement, title: string): HTMLButtonElement {
  const button = document.createElement('button')
  button.className = `pdf-embed-button ${className}`
  button.type = 'button'
  button.title = title
  button.ariaLabel = title
  button.append(icon)
  return button
}

function createPdfIcon(path: string): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  svg.setAttribute('viewBox', '0 0 15 15')
  svg.setAttribute('fill', 'none')
  const pathElement = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  pathElement.setAttribute('d', path)
  pathElement.setAttribute('fill', 'currentColor')
  svg.append(pathElement)
  return svg
}

function createDownloadLink(options: PdfOptions): HTMLAnchorElement {
  const link = document.createElement('a')
  link.className = 'pdf-embed-button pdf-embed-download'
  link.href = options.src
  link.download = options.title
  link.title = 'Download PDF'
  link.ariaLabel = 'Download PDF'
  link.append(createPdfIcon(PDF_ICON_DOWNLOAD))
  return link
}

export function createControls(options: PdfOptions): {
  toolbar: HTMLDivElement
  controls: PdfControls
} {
  const toolbar = document.createElement('div')
  toolbar.className = 'pdf-embed-toolbar'

  const pageGroup = document.createElement('div')
  pageGroup.className = 'pdf-embed-control-group pdf-embed-page-controls'
  const prev = createButton('pdf-embed-prev', createPdfIcon(PDF_ICON_PREV), 'Previous page')
  const pageInput = document.createElement('input')
  pageInput.className = 'pdf-embed-page-input'
  pageInput.type = 'number'
  pageInput.min = '1'
  pageInput.value = String(options.page)
  pageInput.ariaLabel = 'Page'
  const pageTotal = document.createElement('span')
  pageTotal.className = 'pdf-embed-page-total'
  pageTotal.textContent = '/ 0'
  const next = createButton('pdf-embed-next', createPdfIcon(PDF_ICON_NEXT), 'Next page')
  pageGroup.append(prev, pageInput, pageTotal, next)

  const zoomGroup = document.createElement('div')
  zoomGroup.className = 'pdf-embed-control-group pdf-embed-zoom-controls'
  const zoomOut = createButton('pdf-embed-zoom-out', createPdfIcon(PDF_ICON_MINUS), 'Zoom out')
  const zoom = document.createElement('span')
  zoom.className = 'pdf-embed-zoom'
  zoom.textContent = '100%'
  const zoomIn = createButton('pdf-embed-zoom-in', createPdfIcon(PDF_ICON_PLUS), 'Zoom in')
  const download = createDownloadLink(options)
  const fit = document.createElement('select')
  fit.className = 'pdf-embed-fit'
  fit.ariaLabel = 'PDF fit'
  fit.hidden = true
  fit.tabIndex = -1
  fit.setAttribute('aria-hidden', 'true')
  const fitOptions: [PdfFitMode, string][] = [
    ['width', 'Width'],
    ['page', 'Page'],
    ['actual', '100%'],
  ]
  for (const [value, label] of fitOptions) {
    const option = document.createElement('option')
    option.value = value
    option.textContent = label
    fit.appendChild(option)
  }
  fit.value = options.fit === 'custom' ? 'width' : options.fit
  zoomGroup.append(zoomOut, zoom, zoomIn, download, fit)
  toolbar.append(pageGroup, zoomGroup)

  return { toolbar, controls: { pageInput, pageTotal, prev, next, zoom, fit } }
}
