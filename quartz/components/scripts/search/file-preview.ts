import { CODE_FILE_EXTENSIONS } from '../../../util/code-languages'
import { bindClipboardButton } from '../clipboard-button'
import { createPdfEmbed, mountPdfPreview, setPdfEmbedSource } from '../popover/pdf-preview'

export async function renderFilePreview(
  url: URL,
  fileName: string,
  preview: HTMLElement,
  signal: AbortSignal,
  page?: number,
): Promise<boolean> {
  const extension = fileName.split('.').at(-1)?.toLowerCase() ?? ''
  if (extension === 'pdf') {
    const title = fileName.split('/').at(-1)
    // Remounting the previous PDF's embed keeps its toolbar; only the pages reload.
    let pdf = preview.querySelector<HTMLElement>(':scope > .search-pdf-preview > .pdf-embed')
    if (pdf) {
      window.quartzPdfEmbeds?.cleanup(pdf)
      setPdfEmbedSource(pdf, url.toString(), title)
    } else {
      const inner = document.createElement('div')
      inner.className = 'preview-inner search-pdf-preview'
      pdf = createPdfEmbed(url.toString(), title, {
        preload: false,
        annotations: false,
        rootMargin: '0px',
        overscan: 1,
        maxPixelRatio: 2,
        textLayer: true,
      })
      inner.append(pdf)
      preview.replaceChildren(inner)
    }
    if (page && Number.isSafeInteger(page) && page > 0) pdf.dataset.pdfPage = String(page)
    else delete pdf.dataset.pdfPage
    const embed = pdf
    await new Promise<void>((resolve, reject) => {
      mountPdfPreview(
        embed,
        signal,
        resolve,
        () => {
          if (signal.aborted) resolve()
          else reject(new Error('PDF preview could not load.'))
        },
        true,
      )
    })
    return true
  }
  if (!CODE_FILE_EXTENSIONS.has(`.${extension}`)) return false

  // The highlighter chunk loads while the source downloads.
  const highlighter = import('../../../util/highlight-source')
  const response = await fetch(url, { signal, headers: { Accept: 'text/plain, */*;q=0.8' } })
  if (!response.ok) throw new Error(`Source unavailable: ${response.status}`)
  const source = await response.text()
  const { codeAttributes, highlightSourceCode } = await highlighter
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
  for (const [name, value] of Object.entries(codeAttributes(highlighted))) {
    code.setAttribute(name, value)
  }
  code.innerHTML = highlighted.html
  block.append(code)
  caption.append(open, copy)
  figure.append(caption, block)
  inner.append(figure)
  preview.replaceChildren(inner)
  return true
}
