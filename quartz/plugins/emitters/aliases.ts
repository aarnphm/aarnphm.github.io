import { rm } from 'fs/promises'
import path from 'path'
import { VFile } from 'vfile'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx } from '../../util/ctx'
import {
  FilePath,
  FullSlug,
  isRelativeURL,
  joinSegments,
  resolveRelative,
  simplifySlug,
} from '../../util/path'
import { write } from './helpers'

function aliasTargetSlug(file: VFile, aliasTarget: string): FullSlug | undefined {
  const ogSlug = simplifySlug(file.data.slug!)
  const aliasTargetSlug = (
    isRelativeURL(aliasTarget) ? path.normalize(path.join(ogSlug, '..', aliasTarget)) : aliasTarget
  ) as FullSlug

  if (simplifySlug(aliasTargetSlug) === ogSlug) {
    return undefined
  }

  return aliasTargetSlug
}

async function writeAlias(ctx: BuildCtx, file: VFile, aliasTarget: string) {
  const ogSlug = simplifySlug(file.data.slug!)
  const aliasSlug = aliasTargetSlug(file, aliasTarget)
  if (!aliasSlug) return undefined
  const redirUrl = resolveRelative(aliasSlug, ogSlug)
  return write({
    ctx,
    content: `
<!DOCTYPE html>
<html lang="en-us">
<head>
<title>${ogSlug}</title>
<link rel="canonical" href="${redirUrl}">
<meta name="robots" content="noindex">
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=${redirUrl}">
</head>
</html>
        `,
    slug: aliasSlug,
    ext: '.html',
  })
}

async function processFile(ctx: BuildCtx, file: VFile): Promise<FilePath[]> {
  const files = await mapConcurrent(file.data.aliases ?? [], defaultIoConcurrency, aliasTarget =>
    writeAlias(ctx, file, aliasTarget),
  )
  return files.filter(file => file !== undefined)
}

async function deleteAliases(ctx: BuildCtx, file: VFile): Promise<void> {
  for (const aliasTarget of file.data.aliases ?? []) {
    const aliasSlug = aliasTargetSlug(file, aliasTarget)
    if (!aliasSlug) continue
    await rm(joinSegments(ctx.argv.output, `${aliasSlug}.html`) as FilePath, { force: true })
  }
}

function aliasesChanged(file: VFile, previousFile: VFile | undefined): boolean {
  if (!previousFile) return true
  return JSON.stringify(file.data.aliases ?? []) !== JSON.stringify(previousFile.data.aliases ?? [])
}

export const AliasRedirects: QuartzEmitterPlugin = () => ({
  name: 'AliasRedirects',
  async *emit(ctx, content) {
    const files = await mapConcurrent(content, defaultIoConcurrency, ([_tree, file]) =>
      processFile(ctx, file),
    )
    for (const file of files.flat()) {
      yield file
    }
  },
  async *partialEmit(ctx, _content, _resources, changeEvents) {
    for (const changeEvent of changeEvents) {
      if (!changeEvent.file) continue
      if (changeEvent.type === 'delete') {
        await deleteAliases(ctx, changeEvent.file)
        continue
      }
      if (
        changeEvent.type === 'change' &&
        !aliasesChanged(changeEvent.file, changeEvent.previousFile)
      ) {
        continue
      }
      if (changeEvent.type === 'change' && changeEvent.previousFile) {
        await deleteAliases(ctx, changeEvent.previousFile)
      }
      if (changeEvent.type === 'add' || changeEvent.type === 'change') {
        yield* await processFile(ctx, changeEvent.file)
      }
    }
  },
})
