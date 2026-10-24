import { readFile, realpath } from 'node:fs/promises'
import path from 'node:path'
import type { QuartzEmitterPlugin } from '../../types/plugin'
import type { CodeUnit } from '../../util/code-search-types'
import { createMdProcessor } from '../../processors/parse'
import {
  CODE_FILE_EXTENSIONS,
  languageFromExt,
  canonicalCodeLanguage,
} from '../../util/code-languages'
import { notebookTitle } from '../../util/notebook/markdown'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'
import { isFilePath, slugifyFilePath } from '../../util/path'
import { isSearchExcluded } from '../../util/search-exclude'
import { isRecord } from '../../util/type-guards'
import {
  codeSourceId,
  notebookCellRecords,
  type PublishedCodeBlock,
} from '../transformers/code-search-blocks'
import { publishCodeSearch } from './code-search-publication'

function hidden(value: unknown): boolean {
  return (
    isRecord(value) && ['private', 'protected', 'draft', 'noindex'].some(key => value[key] === true)
  )
}

export const CodeSearch: QuartzEmitterPlugin = () => {
  const emit: ReturnType<QuartzEmitterPlugin>['emit'] = async function* (ctx, content) {
    const units: CodeUnit[] = []
    const bodies = new Map<string, string>()
    const root = await realpath(ctx.argv.directory)
    const excluded = (fp: string) =>
      isSearchExcluded(fp) ||
      ctx.cfg.configuration.ignorePatterns.some(pattern => path.matchesGlob(fp, pattern))
    const denied = new Set<string | undefined>(
      content
        .filter(([, file]) => hidden(file.data.frontmatter))
        .map(([, file]) => file.data.relativePath),
    )
    const eligible = (fp: string) =>
      !excluded(fp) && !denied.has(fp) && !path.isAbsolute(fp) && !fp.split('/').includes('..')
    const append = (
      ownerSlug: string,
      title: string,
      ownerPath: string,
      block: PublishedCodeBlock,
      kind: CodeUnit['kind'] = block.kind,
    ) => {
      if (
        block.sourcePath &&
        (!eligible(block.sourcePath) || !ctx.allFiles.some(fp => fp === block.sourcePath))
      )
        return
      const sourceId = codeSourceId(block.text)
      const anchor = kind === 'attachment' ? null : block.anchor
      units.push({
        id: `${ownerSlug}#${anchor ?? 'source'}`,
        ownerSlug,
        title,
        path: block.sourcePath ?? ownerPath,
        kind,
        language: canonicalCodeLanguage(block.originalLanguage),
        originalLanguage: block.originalLanguage,
        sourceId,
        anchor,
        startLine: block.startLine,
        endLine: block.startLine + block.text.split(/\r\n|\n|\r/).length - 1,
      })
      bodies.set(sourceId, block.text)
    }
    for (const [, file] of content) {
      const fp = file.data.relativePath
      const slug = file.data.slug
      if (!fp || !slug || !eligible(fp) || hidden(file.data.frontmatter)) continue
      for (const block of file.data.codeSearchBlocks ?? [])
        append(slug, file.data.frontmatter?.title ?? slug, fp, block)
    }
    let parser: ReturnType<typeof createMdProcessor> | undefined
    for (const fp of [...ctx.allFiles].sort()) {
      if (!eligible(fp)) continue
      const ext = path.extname(fp).toLowerCase()
      if (ext !== '.ipynb' && (ext === '.json' || !CODE_FILE_EXTENSIONS.has(ext))) continue
      const target = await realpath(path.join(root, fp))
      const relative = path.relative(root, target)
      if (relative.startsWith('..') || path.isAbsolute(relative) || !eligible(relative)) continue
      const text = await readFile(target, 'utf8')
      if (text.includes('\0')) continue
      if (ext === '.ipynb') {
        const notebook = parseNotebookDoc(text, fp)
        if (isNotebookParseError(notebook))
          throw new Error(`Cannot index notebook ${fp}: ${notebook.reason}`)
        if (hidden(notebook.metadata) || hidden(notebook.metadata.quartz)) continue
        const md = (parser ??= createMdProcessor(ctx))
        const slug = slugifyFilePath(fp, true)
        for (const cell of notebook.cells) {
          for (const block of notebookCellRecords(cell, source => md.parse(source)))
            append(slug, notebookTitle(notebook, fp), fp, block)
        }
      } else {
        if (!isFilePath(fp)) continue
        const slug = slugifyFilePath(fp)
        append(
          slug,
          path.basename(fp),
          fp,
          {
            text,
            originalLanguage: languageFromExt(ext) ?? 'text',
            anchor: '',
            startLine: 1,
            kind: 'block',
          },
          'attachment',
        )
      }
    }
    yield* publishCodeSearch(ctx, units, bodies)
  }
  return { name: 'CodeSearch', emit, partialEmit: emit }
}
