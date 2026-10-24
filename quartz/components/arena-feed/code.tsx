import type { RefObject } from 'preact'
import { useEffect, useState } from 'preact/hooks'
import type { ArenaReaderArtifact } from '../../util/arena-reader'

type Highlighted = { attributes: Record<string, string>; html: string }

export function CodeContent({
  artifact,
  contentRef,
}: {
  artifact: Extract<ArenaReaderArtifact, { kind: 'code' }>
  contentRef: RefObject<HTMLDivElement>
}) {
  // content.tsx reaches the PDF reader and notes bundles too, so the Wasm highlighter loads on demand.
  const [highlighted, setHighlighted] = useState<Highlighted | null>(null)
  useEffect(() => {
    let live = true
    setHighlighted(null)
    void import('../../util/highlight-source').then(
      ({ codeAttributes, highlightSourceCode }) => {
        if (!live) return
        const result = highlightSourceCode(artifact.code, artifact.fileName)
        setHighlighted({ attributes: codeAttributes(result), html: result.html })
      },
      () => {},
    )
    return () => {
      live = false
    }
  }, [artifact.code, artifact.fileName])
  return (
    <div ref={contentRef} class="arena-reader-prose arena-reader-code">
      <figure>
        <figcaption>{artifact.fileName}</figcaption>
        <pre tabIndex={0} aria-label={`Source code: ${artifact.fileName}`}>
          {highlighted ? (
            <code
              {...highlighted.attributes}
              dangerouslySetInnerHTML={{ __html: highlighted.html }}
            />
          ) : (
            <code>{artifact.code}</code>
          )}
        </pre>
      </figure>
    </div>
  )
}
