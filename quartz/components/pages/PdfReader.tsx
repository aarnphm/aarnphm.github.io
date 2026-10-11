import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { pdfResourceOf } from '../../util/course'
import { scriptJson } from '../../util/pdf-marks'
// @ts-ignore
import script from '../scripts/pdf-reader.inline'
import style from '../styles/pdf-reader.scss'

export default (() => {
  const PdfReader: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    const data = fileData.pdfReader
    const resource = pdfResourceOf(fileData)
    // R2: a course page whose PDF is not in the manifest keeps the strip and explains itself.
    if (resource && !data) {
      return (
        <article class="pdf-reader" data-pdf-reader>
          <div class="pdf-reader-mount">
            <p class="pdf-reader-status" role="status">
              This PDF is not in the build: <code>{resource.record.file}</code> was not found under
              the course directory.
            </p>
          </div>
        </article>
      )
    }
    return (
      <article class="pdf-reader" data-pdf-reader>
        {/* Resource pages inline their record; the worker swaps `null` on the shared shell. */}
        <script
          type="application/json"
          id="pdf-reader-data"
          dangerouslySetInnerHTML={{ __html: data ? scriptJson(data) : 'null' }}
        />
        <div class="pdf-reader-mount" data-pdf-reader-mount>
          <p class="pdf-reader-status" role="status">
            Loading the reader…
          </p>
        </div>
        <noscript>
          The reader needs JavaScript to render pages and marks.{' '}
          {data?.slug ? (
            <a href={`/${encodeURI(data.slug)}?raw=1`}>Open the PDF directly.</a>
          ) : (
            <a href="/">Back to the garden.</a>
          )}
        </noscript>
      </article>
    )
  }
  PdfReader.css = style
  PdfReader.afterDOMLoaded = script
  return PdfReader
}) satisfies QuartzComponentConstructor
