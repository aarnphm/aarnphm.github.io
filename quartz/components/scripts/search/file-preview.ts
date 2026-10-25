import { CODE_FILE_EXTENSIONS } from '../../../util/code-languages'
import { bindClipboardButton } from '../clipboard-button'
import { renderPdfPreview } from './pdf-preview-cache'

export async function renderFilePreview(
  url: URL,
  fileName: string,
  preview: HTMLElement,
  signal: AbortSignal,
  page?: number,
  query = '',
): Promise<boolean> {
  const extension = fileName.split('.').at(-1)?.toLowerCase() ?? ''
  if (extension === 'pdf') {
    await renderPdfPreview(
      preview,
      url,
      fileName.split('/').at(-1) ?? fileName,
      signal,
      page,
      query,
    )
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
