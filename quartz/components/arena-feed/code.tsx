import type { RefObject } from 'preact'
import { useMemo } from 'preact/hooks'
import type { ArenaReaderArtifact } from '../../util/arena-reader'
import { highlightSourceCode } from '../../util/highlight-source'

export function CodeContent({
  artifact,
  contentRef,
}: {
  artifact: Extract<ArenaReaderArtifact, { kind: 'code' }>
  contentRef: RefObject<HTMLDivElement>
}) {
  const highlighted = useMemo(
    () => highlightSourceCode(artifact.code, artifact.fileName),
    [artifact.code, artifact.fileName],
  )
  return (
    <div ref={contentRef} class="arena-reader-prose arena-reader-code">
      <figure>
        <figcaption>{artifact.fileName}</figcaption>
        <pre tabIndex={0} aria-label={`Source code: ${artifact.fileName}`}>
          <code
            data-language={highlighted.language}
            dangerouslySetInnerHTML={{ __html: highlighted.html }}
          />
        </pre>
      </figure>
    </div>
  )
}
