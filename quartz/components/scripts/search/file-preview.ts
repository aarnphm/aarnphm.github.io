import { CODE_FILE_EXTENSIONS } from '../../../util/code-languages'
import { bindClipboardButton } from '../clipboard-button'
import { createPdfEmbed, mountPdfPreview } from '../popover/pdf-preview'

export async function renderFilePreview(
  url: URL,
  fileName: string,
  preview: HTMLElement,
  signal: AbortSignal,
  page?: number,
): Promise<boolean> {
  const extension = fileName.split('.').at(-1)?.toLowerCase() ?? ''
  if (extension === 'pdf') {
    const inner = document.createElement('div')
    inner.className = 'preview-inner search-pdf-preview'
    const pdf = createPdfEmbed(url.toString(), fileName.split('/').at(-1), {
      preload: false,
      annotations: false,
      rootMargin: '0px',
      overscan: 1,
      maxPixelRatio: 2,
      textLayer: true,
    })
    if (page && Number.isSafeInteger(page) && page > 0) pdf.dataset.pdfPage = String(page)
    inner.append(pdf)
    preview.replaceChildren(inner)
    await new Promise<void>((resolve, reject) => {
      mountPdfPreview(pdf, signal, resolve, () => {
        if (signal.aborted) resolve()
        else reject(new Error('PDF preview could not load.'))
      })
    })
    return true
  }
  if (!CODE_FILE_EXTENSIONS.has(`.${extension}`)) return false

  const loading = document.createElement('p')
  loading.setAttribute('role', 'status')
  loading.textContent = 'loading source…'
  preview.replaceChildren(loading)
  const response = await fetch(url, { signal, headers: { Accept: 'text/plain, */*;q=0.8' } })
  if (!response.ok) throw new Error(`Source unavailable: ${response.status}`)
  const source = await response.text()
  if (signal.aborted) return true
  const { highlightSourceCode } = await import('../../../util/highlight-source')
  if (signal.aborted) return true
  const highlighted = highlightSourceCode(source, fileName)
  const inner = document.createElement('div')
  inner.className = 'preview-inner search-code-preview'
  const figure = document.createElement('figure')
  const caption = document.createElement('figcaption')
  const open = document.createElement('a')
  open.href = url.toString()
  open.textContent = fileName.split('/').at(-1) ?? fileName
  open.title = 'open source file'
  const copy = document.createElement('button')
  copy.type = 'button'
  copy.textContent = 'copy'
  copy.setAttribute('aria-label', 'copy source')
  const disposeCopy = bindClipboardButton(copy, () => source)
  signal.addEventListener('abort', disposeCopy, { once: true })
  const block = document.createElement('pre')
  block.tabIndex = 0
  block.setAttribute('aria-label', `Source code: ${fileName}`)
  const code = document.createElement('code')
  code.dataset.language = highlighted.language
  code.innerHTML = highlighted.html
  block.append(code)
  caption.append(open, copy)
  figure.append(caption, block)
  inner.append(figure)
  preview.replaceChildren(inner)
  return true
}
