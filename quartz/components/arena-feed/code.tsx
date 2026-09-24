import type { RefObject } from 'preact'
import hljs from 'highlight.js/lib/common'
import { useMemo } from 'preact/hooks'
import type { ArenaReaderArtifact } from '../../util/arena-reader'
import { escapeHTML } from '../../util/escape'

export function highlightArenaSource(
  code: string,
  fileName: string,
): { html: string; language: string } {
  const name = fileName.toLowerCase()
  const special: Record<string, string> = {
    dockerfile: 'dockerfile',
    makefile: 'makefile',
    '.bashrc': 'bash',
    '.zshrc': 'bash',
  }
  const extension = name.split('.').at(-1) ?? ''
  const aliases: Record<string, string> = {
    cu: 'cpp',
    cuh: 'cpp',
    pyi: 'python',
    jsx: 'javascript',
    tsx: 'typescript',
    ipynb: 'json',
    toml: 'ini',
  }
  const candidate = Object.hasOwn(special, name)
    ? special[name]
    : Object.hasOwn(aliases, extension)
      ? aliases[extension]
      : extension
  const language = hljs.getLanguage(candidate) ? candidate : 'plaintext'
  // Large files remain complete without running a tokenizer on the UI thread.
  const html =
    code.length <= 200_000 && language !== 'plaintext'
      ? hljs.highlight(code, { language, ignoreIllegals: true }).value
      : escapeHTML(code)
  return { html, language }
}

export function CodeContent({
  artifact,
  contentRef,
}: {
  artifact: Extract<ArenaReaderArtifact, { kind: 'code' }>
  contentRef: RefObject<HTMLDivElement>
}) {
  const highlighted = useMemo(
    () => highlightArenaSource(artifact.code, artifact.fileName),
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
