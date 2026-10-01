import type { QuartzComponent, QuartzComponentConstructor } from '../../types/component'
// @ts-ignore
import script from '../scripts/pdf-reader.inline'
import style from '../styles/pdf-reader.scss'

export default (() => {
  const PdfReader: QuartzComponent = () => (
    <article class="pdf-reader" data-pdf-reader>
      {/* The worker swaps `null` for the document record; the client reads it on nav. */}
      <script
        type="application/json"
        id="pdf-reader-data"
        dangerouslySetInnerHTML={{ __html: 'null' }}
      />
      <div class="pdf-reader-mount" data-pdf-reader-mount>
        <p class="pdf-reader-status" role="status">
          Loading the reader…
        </p>
      </div>
      <noscript>
        The reader needs JavaScript to render pages and marks. <a href="/">Back to the garden.</a>
      </noscript>
    </article>
  )
  PdfReader.css = style
  PdfReader.afterDOMLoaded = script
  return PdfReader
}) satisfies QuartzComponentConstructor
