import { Root as HTMLRoot } from 'hast'
import { toString } from 'hast-util-to-string'
import readingTime, { ReadTimeResults } from 'reading-time'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { renderDescriptionHtml, resolveManifestDescription } from '../../util/description'
import { descriptionManifestPath, loadDescriptionManifest } from '../../util/description-manifest'
import { escapeHTML } from '../../util/escape'
import { releasedOnReload } from '../../util/module-caches'
import { simplifySlug, type FullSlug, type SimpleSlug } from '../../util/path'
import { extractWikilinks, resolveWikilinkTarget } from '../../util/wikilinks'

export interface Options {
  descriptionLength: number
  maxDescriptionLength: number
  replaceExternalLinks: boolean
}

const defaultOptions: Options = {
  descriptionLength: 150,
  maxDescriptionLength: 300,
  replaceExternalLinks: true,
}

const urlRegex = new RegExp(
  /(https?:\/\/)?(?<domain>([\da-z.-]+)\.([a-z.]{2,6})(:\d+)?)(?<path>[/\w.-]*)(\?[/\w.=&;-]*)?/,
  'g',
)

const MAX_TEXT_CACHE_ENTRIES = 512
const textCache = releasedOnReload(
  new Map<string, { text: string; readingTime: ReadTimeResults }>(),
)

function markdownBody(source: string): string {
  const frontmatter = /^---\r?\n[\s\S]*?\r?\n---\r?\n?/.exec(source)
  return frontmatter ? source.slice(frontmatter[0].length) : source
}

function sourceFingerprint(source: string): string {
  const body = markdownBody(source)
  let hash = 0x811c9dc5
  for (let i = 0; i < body.length; i += 1) {
    hash ^= body.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return `${body.length}:${hash}`
}

function textCacheKey(fileValue: unknown, slug: FullSlug): string | undefined {
  const source = typeof fileValue === 'string' ? fileValue : fileValue?.toString()
  return source ? `${slug}:${sourceFingerprint(source)}` : undefined
}

function setTextCache(key: string, value: { text: string; readingTime: ReadTimeResults }): void {
  textCache.set(key, value)
  if (textCache.size <= MAX_TEXT_CACHE_ENTRIES) return
  const first = textCache.keys().next().value
  if (first) textCache.delete(first)
}

export const Description: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: 'Description',
    fileDependencies: (ctx, file) =>
      !file ||
      file.data.bases ||
      file.data.jsonCanvas ||
      file.data.hasExternalDescription ||
      !file.data.frontmatter?.description
        ? [descriptionManifestPath(ctx)]
        : [],
    htmlPlugins(ctx) {
      return [
        () => {
          return async (tree: HTMLRoot, file) => {
            const currentSlug = file.data.slug as FullSlug
            const descriptionLinks: Set<SimpleSlug> = new Set()

            const rawFrontMatterDescription =
              typeof file.data.frontmatter?.description === 'string' &&
              file.data.frontmatter.description.trim().length > 0
                ? file.data.frontmatter.description
                : undefined
            const manifestDescription =
              rawFrontMatterDescription === undefined
                ? resolveManifestDescription(
                    await loadDescriptionManifest(ctx),
                    currentSlug,
                    file.data.relativePath,
                  )
                : undefined
            const authoredDescription = rawFrontMatterDescription ?? manifestDescription
            file.data.hasExternalDescription = Boolean(manifestDescription)
            if (manifestDescription && file.data.frontmatter) {
              file.data.frontmatter.description = manifestDescription
            }

            // Extract and track wikilinks from authored description
            if (typeof authoredDescription === 'string') {
              const wikilinks = extractWikilinks(authoredDescription)
              for (const link of wikilinks) {
                const resolved = resolveWikilinkTarget(link, currentSlug)
                if (resolved) {
                  descriptionLinks.add(simplifySlug(resolved.slug))
                }
              }
            }

            const processDescription = (desc: string): string => {
              const sentences = desc.replace(/\s+/g, ' ').split(/\.\s/)
              let finalDesc = ''
              let sentenceIdx = 0

              // Add full sentences until we exceed the guideline length
              while (sentenceIdx < sentences.length) {
                const sentence = sentences[sentenceIdx]
                if (!sentence) break

                const currentSentence = sentence.endsWith('.') ? sentence : sentence + '.'
                const nextLength = finalDesc.length + currentSentence.length + (finalDesc ? 1 : 0)

                // Add the sentence if we're under the guideline length
                // or if this is the first sentence (always include at least one)
                if (nextLength <= opts.descriptionLength || sentenceIdx === 0) {
                  finalDesc += (finalDesc ? ' ' : '') + currentSentence
                  sentenceIdx++
                } else {
                  break
                }
              }
              return finalDesc.length > opts.maxDescriptionLength
                ? finalDesc.slice(0, opts.maxDescriptionLength) + '...'
                : finalDesc
            }

            const processedAuthoredDesc = authoredDescription
              ? renderDescriptionHtml(authoredDescription, currentSlug)
              : undefined

            const isStructuredDocument = Boolean(file.data.bases || file.data.jsonCanvas)
            const precomputedText =
              isStructuredDocument && typeof file.data.text === 'string'
                ? file.data.text
                : undefined

            const cacheKey =
              precomputedText === undefined ? textCacheKey(file.value, currentSlug) : undefined
            const cachedText = cacheKey ? textCache.get(cacheKey) : undefined
            let text = precomputedText ?? cachedText?.text
            let textReadingTime = cachedText?.readingTime
            if (text === undefined || !textReadingTime) {
              if (text === undefined) {
                text = escapeHTML(toString(tree))
                if (opts.replaceExternalLinks) {
                  text = text.replace(urlRegex, '$<domain>' + '$<path>')
                }
              }
              textReadingTime = readingTime(text)
              if (cacheKey) {
                setTextCache(cacheKey, { text, readingTime: textReadingTime })
              }
            }

            // Authored descriptions stay intact; length limits apply to generated body excerpts.
            const processedPlainDesc = isStructuredDocument ? '' : processDescription(text)

            file.data.description = processedAuthoredDesc || processedPlainDesc || ''
            file.data.rawDescription = authoredDescription || processedPlainDesc || ''

            // Process abstract with wikilinks support
            let abstractText = file.data.frontmatter?.abstract
            if (abstractText) {
              // Extract and track wikilinks from abstract
              const abstractWikilinks = extractWikilinks(abstractText)
              for (const link of abstractWikilinks) {
                const resolved = resolveWikilinkTarget(link, currentSlug)
                if (resolved) {
                  descriptionLinks.add(simplifySlug(resolved.slug))
                }
              }
              abstractText = renderDescriptionHtml(abstractText, currentSlug)
            }

            file.data.abstract = abstractText ?? processedPlainDesc
            file.data.text = text
            file.data.readingTime = textReadingTime

            // Merge description links with existing links
            if (descriptionLinks.size > 0) {
              const existingLinks = file.data.links || []
              file.data.links = [...new Set([...existingLinks, ...descriptionLinks])]
            }
          }
        },
      ]
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    description: string
    rawDescription: string
    hasExternalDescription?: boolean
    abstract: string
    text: string
    readingTime: ReadTimeResults
  }
}
