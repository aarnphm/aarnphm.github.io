import path from 'path'
import { VFile } from 'vfile'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx } from '../../util/ctx'
import { FilePath, FullSlug, isRelativeURL, resolveRelative, simplifySlug } from '../../util/path'
import { ProcessedContent } from '../vfile'
import { write, removeWritten } from './helpers'

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

function publicPath(slug: FullSlug): string {
  const simplified = simplifySlug(slug)
  return simplified === '/' ? '/' : `/${simplified}`
}

function canonicalMarkdownPath(file: VFile): string {
  return file.data.slug === 'index' ? '/llms.txt' : `/${file.data.slug}.md`
}

export function aliasRedirectRules(content: ProcessedContent[]): string {
  const rules = new Map<string, string>()
  const addRule = (source: string, destination: string) => {
    const existing = rules.get(source)
    if (existing && existing !== destination) {
      throw new Error(`conflicting alias redirect for ${source}`)
    }
    rules.set(source, destination)
  }

  for (const [_tree, file] of content) {
    const canonicalHtml = publicPath(file.data.slug!)
    const canonicalMarkdown = canonicalMarkdownPath(file)
    for (const aliasTarget of file.data.aliases ?? []) {
      const aliasSlug = aliasTargetSlug(file, aliasTarget)
      if (!aliasSlug) continue
      const aliasHtml = publicPath(aliasSlug)
      addRule(aliasHtml, canonicalHtml)
      addRule(`${aliasHtml}.md`, canonicalMarkdown)
    }
  }

  if (rules.size > 2000) {
    throw new Error(`static alias redirects exceed the 2,000 rule limit: ${rules.size}`)
  }

  return [...rules]
    .sort(([sourceA], [sourceB]) => sourceA.localeCompare(sourceB))
    .map(([source, destination]) => `${source} ${destination} 308`)
    .join('\n')
}

function writeRedirectRules(ctx: BuildCtx, content: ProcessedContent[]): Promise<FilePath> {
  return write({ ctx, content: aliasRedirectRules(content), slug: '_redirects', ext: '' })
}

async function deleteAliases(ctx: BuildCtx, file: VFile): Promise<void> {
  for (const aliasTarget of file.data.aliases ?? []) {
    const aliasSlug = aliasTargetSlug(file, aliasTarget)
    if (!aliasSlug) continue
    await removeWritten(ctx, aliasSlug, '.html')
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
    yield writeRedirectRules(ctx, content)
  },
  async *partialEmit(ctx, content, _resources, changeEvents) {
    let redirectsChanged = false
    for (const changeEvent of changeEvents) {
      if (!changeEvent.file) continue
      if (changeEvent.type === 'delete') {
        await deleteAliases(ctx, changeEvent.file)
        redirectsChanged ||= (changeEvent.file.data.aliases?.length ?? 0) > 0
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
        redirectsChanged ||=
          (changeEvent.file.data.aliases?.length ?? 0) > 0 ||
          (changeEvent.previousFile?.data.aliases?.length ?? 0) > 0
      }
    }
    if (redirectsChanged) yield writeRedirectRules(ctx, content)
  },
})
