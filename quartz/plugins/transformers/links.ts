import { Element } from 'hast'
import { h } from 'hastscript'
import isAbsoluteUrl from 'is-absolute-url'
import path from 'path'
import { visit } from 'unist-util-visit'
import type { QuartzTransformerPlugin } from '../../types/plugin'
import type { ReferenceCatalog } from '../../util/references/catalog-types'
import type { ArenaBlock } from './arena'
import { FullSlug, SimpleSlug, TransformOptions, isFullSlug, simplifySlug } from '../../util/path'
import { referenceCatalogFor } from '../../util/references/context'
import { attachHastReferences } from '../../util/references/hast'
import { referenceResolutionSignature } from '../../util/references/markdown'
import { transformResourceUrl } from '../../util/resource-url'
import { decorateLink } from './links/decorate'
import { metadataDisablesPopover, readLinkMetadata } from './links/metadata'
import { canonicalResourceUrl, resolveInternalLink } from './links/resolve'

interface Options {
  enableArxivEmbed: boolean
  enableRawEmbed: boolean
  enableIndicatorHook: boolean
  /** How to resolve Markdown paths */
  markdownLinkResolution: TransformOptions['strategy']
  /** Strips folders from a link so that it looks nice */
  prettyLinks: boolean
  openLinksInNewTab: boolean
  lazyLoad: boolean
  externalLinkIcon: boolean
}

const defaultOptions: Options = {
  enableArxivEmbed: false,
  enableRawEmbed: false,
  enableIndicatorHook: true,
  markdownLinkResolution: 'absolute',
  prettyLinks: true,
  openLinksInNewTab: false,
  lazyLoad: false,
  externalLinkIcon: true,
}

// ofm's rehype-raw pass re-parses notes containing raw HTML and camelCases data attributes, so
// `data-*` properties arrive under either spelling depending on the note.
const PDF_SRC_PROPERTIES = ['data-pdf-src', 'dataPdfSrc']

function transformResourceProperty(
  node: Element,
  property: string,
  fileSlug: FullSlug,
  transformOptions: TransformOptions,
) {
  const value = node.properties[property]
  if (typeof value !== 'string') return
  node.properties[property] =
    canonicalResourceUrl(node, value, fileSlug) ??
    transformResourceUrl(fileSlug, value, transformOptions)
}

export const CrawlLinks: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: 'LinkProcessing',
    htmlPlugins(ctx) {
      const { cfg } = ctx
      let baseCatalog: ReferenceCatalog | undefined
      let baseSlugs = new Set<string>()
      const basePages = () => {
        const catalog = referenceCatalogFor(ctx)
        if (catalog !== baseCatalog) {
          baseCatalog = catalog
          baseSlugs = new Set(
            catalog.entries.filter(entry => entry.kind === 'base').map(entry => entry.slug),
          )
        }
        return baseSlugs
      }
      return [
        () => {
          return (tree, file) => {
            attachHastReferences(tree, file)
            const fileSlug = file.data.slug
            if (!fileSlug) return
            const outgoing: Set<SimpleSlug> = new Set(file.data.jsonCanvas ? file.data.links : [])

            const transformOptions: TransformOptions = {
              strategy: opts.markdownLinkResolution,
              allSlugs: ctx.allSlugs,
            }

            const shouldRewriteLinks = ({ tagName, properties }: Element) =>
              tagName === 'a' && Boolean(properties.href) && typeof properties.href === 'string'

            visit(tree, 'element', (node: Element) => {
              if (!shouldRewriteLinks(node)) {
                return
              }
              if (typeof node.properties.href !== 'string') return
              let dest = node.properties.href
              const ext: string = path.extname(dest).toLowerCase()
              const request = node.data?.referenceRequest
              const metadata = request
                ? request.resolvedSettings?.kind === 'object'
                  ? request.resolvedSettings.value
                  : undefined
                : readLinkMetadata(node.properties['data-metadata'] ?? node.properties.dataMetadata)
              const hasProtocol = /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(dest)
              const decoration = decorateLink(node, dest, ext, metadata, {
                ...opts,
                baseUrl: cfg.configuration.baseUrl,
                fileSlug: file.data.slug,
                email: file.data.frontmatter?.email,
                basePages,
              })
              dest = decoration.destination

              if ((decoration.isExternal && opts.openLinksInNewTab) || ['.ipynb'].includes(ext)) {
                node.properties.target = '_blank'
              }

              // don't process external links, protocol URLs, or intra-document anchors
              const isInternal = !(
                isAbsoluteUrl(dest, { httpOnly: false }) ||
                dest.startsWith('#') ||
                hasProtocol
              )

              if (isInternal && metadataDisablesPopover(metadata)) {
                node.properties['data-no-popover'] = true
              }

              const canonical =
                !hasProtocol && !isAbsoluteUrl(dest, { httpOnly: false })
                  ? resolveInternalLink(ctx, file, node, dest)
                  : undefined
              if (canonical) {
                dest = node.properties.href = canonical.href
                if (isInternal) {
                  outgoing.add(simplifySlug(canonical.slug))
                  node.properties['data-slug'] = canonical.slug
                }
              }

              // rewrite link internals if prettylinks is on
              if (
                opts.prettyLinks &&
                isInternal &&
                !(
                  Array.isArray(node.properties.className) &&
                  node.properties.className.includes('tag-link')
                ) &&
                node.children.length === 1 &&
                node.children[0].type === 'text' &&
                !node.children[0].value.startsWith('#')
              ) {
                node.children[0].value = path.basename(node.children[0].value)
              }

              // add indicator hook after handling all prettyLinks, inspired by gwern
              if (opts.enableIndicatorHook) {
                node.children = [h('span.indicator-hook'), ...node.children]
              }
            })

            const shouldTransformResources = ({ tagName, properties }: Element) =>
              (['img', 'video', 'audio', 'iframe'].includes(tagName) &&
                Boolean(properties.src) &&
                typeof properties.src === 'string') ||
              (tagName === 'div' &&
                PDF_SRC_PROPERTIES.some(
                  property =>
                    Boolean(properties[property]) && typeof properties[property] === 'string',
                ))

            visit(tree, 'element', node => {
              if (!shouldTransformResources(node)) return
              if (opts.lazyLoad) node.properties.loading = 'lazy'
              for (const property of ['src', ...PDF_SRC_PROPERTIES]) {
                transformResourceProperty(node, property, fileSlug, transformOptions)
              }
            })

            const fmLinks = file.data.frontmatterLinks
            if (fmLinks) {
              for (const links of Object.values(fmLinks)) {
                for (const link of links) {
                  outgoing.add(simplifySlug(link.slug))
                }
              }
            }

            const arenaData = file.data.arenaData
            if (arenaData) {
              const collectArenaInternalLinks = (blocks: ArenaBlock[]): void => {
                for (const block of blocks) {
                  if (block.internalSlug && isFullSlug(block.internalSlug)) {
                    outgoing.add(simplifySlug(block.internalSlug))
                  }
                  if (block.subItems) {
                    collectArenaInternalLinks(block.subItems)
                  }
                }
              }
              for (const channel of arenaData.channels) {
                collectArenaInternalLinks(channel.blocks)
              }
            }

            file.data.links = [...outgoing]
            file.data.referenceResolutionSignature = referenceResolutionSignature(ctx, file)
          }
        },
      ]
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    links: SimpleSlug[]
  }
}
