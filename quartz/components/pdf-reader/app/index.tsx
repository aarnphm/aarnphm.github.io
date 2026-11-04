import { useState } from 'preact/hooks'
import type { IndexDocument } from './data'

export function ReaderIndex({ documents }: { documents: IndexDocument[] }) {
  const [filter, setFilter] = useState('')
  const query = filter.trim().toLowerCase()
  const visible = documents.filter(
    doc =>
      !query || doc.title.toLowerCase().includes(query) || doc.slug.toLowerCase().includes(query),
  )
  return (
    <div class="pdf-reader-library" data-keyboard-scroll-scope data-keyboard-scroll>
      <header>
        <h1>reader</h1>
        <p>Open any PDFs to read with annotations.</p>
        <input
          type="search"
          value={filter}
          placeholder="filter by title or path"
          aria-label="Filter documents"
          onInput={event => setFilter(event.currentTarget.value)}
        />
      </header>
      {visible.length === 0 ? (
        <p class="pdf-reader-empty">No documents match.</p>
      ) : (
        <ol class="pdf-reader-library-list">
          {visible.map(doc => (
            <li key={doc.slug}>
              <a href={encodeURI(doc.readPath)}>
                <span class="pdf-reader-library-title">{doc.title}</span>
                <span class="pdf-reader-library-meta">
                  {doc.slug} · cited by {doc.citations}
                </span>
              </a>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
