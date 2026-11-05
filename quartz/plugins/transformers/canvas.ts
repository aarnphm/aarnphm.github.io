import type { Element } from 'hast'
import path from 'path'
import { visit } from 'unist-util-visit'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { FullSlug, isFullSlug, resolveRelative, simplifySlug } from '../../util/path'
import { referenceCatalogFor } from '../../util/references/context'
import { extractLiteralReferences } from '../../util/references/literal'
import { referenceFields, referenceRequest } from '../../util/references/request'
import { missingReferenceRoute, referenceRoute } from '../../util/references/route'
import { referenceFromParts } from '../../util/references/syntax'
import { interpretReference } from '../../util/references/target'
import { parseJsonCanvas } from './jcast'
import { JcastCanvas, JcastResolvedWikilink, isJcastCanvasNode } from './jcast/types'
import { visitJcast } from './jcast/visitor'

function stringProperty(node: Element, ...names: string[]): string | undefined {
  for (const name of names) {
    const value = node.properties[name]
    if (typeof value === 'string') return value
  }
  return undefined
}

export const JsonCanvas: QuartzTransformerPlugin = () => {
  return {
    name: 'JsonCanvas',
    markdownPlugins() {
      return [
        () => {
          return (tree, file) => {
            const isCanvasFile = file.path?.endsWith('.canvas')

            if (!isCanvasFile) {
              file.data.jsonCanvas = false
              return
            }

            // Mark as bases file
            file.data.jsonCanvas = true
            // clear tree to prevent markdown parsing of JSON content
            tree.children = []
          }
        },
      ]
    },
    htmlPlugins(ctx) {
      return [
        () => async (tree, file) => {
          const slug = file.data.slug as FullSlug

          if (file.data.jsonCanvas) {
            try {
              const canvasContent = String(file.value)
              const jcast = await processCanvasFile(canvasContent, ctx, slug)

              // store raw content for emitter to copy to output
              file.data.canvasContent = canvasContent

              // store processed canvas data
              file.data.canvas = jcast

              // extract links for graph
              const linkedSlugs: string[] = []
              for (const [, node] of jcast.data.nodeMap) {
                if (node.data?.resolved?.slug) {
                  linkedSlugs.push(node.data.resolved.slug)
                }
                if (node.data?.wikilinks) {
                  for (const link of node.data.wikilinks) {
                    if (link.resolvedSlug) {
                      linkedSlugs.push(link.resolvedSlug)
                    }
                  }
                }
              }
              file.data.links = linkedSlugs.filter(isFullSlug).map(simplifySlug)

              // extract searchable text
              const textContent: string[] = []
              const fileNodes: string[] = []
              for (const [, node] of jcast.data.nodeMap) {
                const canvasNode = node.data?.canvas
                if (!canvasNode) continue

                if (canvasNode.type === 'text' && canvasNode.text) {
                  textContent.push(canvasNode.text)
                } else if (canvasNode.type === 'file' && canvasNode.file) {
                  fileNodes.push(canvasNode.file)
                }
              }

              const searchableContent = [
                ...textContent,
                ...fileNodes.map(f => path.basename(f, '.md')),
              ].join(' ')

              file.data.text = searchableContent
            } catch (error) {
              console.error(`Failed to process canvas ${file.data.slug}:`, error)
            }
            return
          }

          visit(tree, 'element', node => {
            const marker = node.properties.dataCanvasEmbed ?? node.properties['data-canvas-embed']
            if (marker === true || marker === '' || marker === 'true') {
              try {
                const canvasFile = stringProperty(node, 'dataCanvasFile', 'data-canvas-file')
                const canvasTitle =
                  stringProperty(node, 'dataCanvasTitle', 'data-canvas-title') || canvasFile
                const canvasSlug =
                  node.data?.referenceRequest?.route?.slug ??
                  stringProperty(node, 'dataSlug', 'data-slug')

                // strip .canvas extension for slug and link
                const linkHref = canvasSlug || canvasFile?.replace(/\.canvas$/, '') || ''
                const resourceBase = linkHref.startsWith('/') ? linkHref : `/${linkHref}`

                // embed configuration: static preview, no interaction
                const embedConfig = { zoom: false, showPreviewOnHover: false }

                // transform into full canvas embed container
                node.tagName = 'div'
                node.properties = {
                  class: 'canvas-embed-container',
                  'data-canvas': `${resourceBase}.canvas`,
                  'data-meta': `${resourceBase}.meta.json`,
                  'data-cfg': JSON.stringify(embedConfig),
                  'data-canvas-title': canvasTitle,
                  'data-embed': 'true',
                  'data-navigate-to': linkHref,
                  style:
                    'position: relative; width: 100%; height: clamp(400px, 50vh, 800px); cursor: pointer; margin: 2rem auto;',
                }

                // loading placeholder
                node.children = [
                  {
                    type: 'element',
                    tagName: 'div',
                    properties: { class: 'canvas-loading' },
                    children: [{ type: 'text', value: 'loading canvas...' }],
                  },
                ]
              } catch (error) {
                const target =
                  stringProperty(node, 'dataCanvasFile', 'data-canvas-file') ||
                  stringProperty(node, 'dataCanvasTitle', 'data-canvas-title')
                console.error(`Failed to embed canvas ${target}:`, error)
                node.tagName = 'div'
                node.properties = {
                  class: 'canvas-embed-error',
                  style:
                    'color: var(--gray); padding: 1em; border: 1px solid var(--lightgray); border-radius: 4px;',
                }
                node.children = [{ type: 'text', value: `Failed to load canvas: ${target}` }]
              }
            }
          })
        },
      ]
    },
  }
}

interface CanvasNodeMeta {
  slug?: string
  href?: string
  displayName?: string
  description?: string
  content?: string
  resolvedText?: string
  wikilinks?: { resolvedSlug?: string; resolvedHref?: string; missing: boolean }[]
}

export function collectCanvasMeta(jcast: JcastCanvas): Record<string, CanvasNodeMeta> {
  const meta: Record<string, CanvasNodeMeta> = {}
  for (const [id, node] of jcast.data.nodeMap) {
    if (!node.data) {
      continue
    }

    const entry: CanvasNodeMeta = {}

    if (node.data.resolved) {
      entry.slug = node.data.resolved.slug
      entry.href = node.data.resolved.href
      entry.displayName = node.data.resolved.displayName
      if (node.data.resolved.description) {
        entry.description = node.data.resolved.description
      }
      if (node.data.resolved.content) {
        entry.content = node.data.resolved.content
      }
    }

    if (node.data.resolvedText) {
      entry.resolvedText = node.data.resolvedText
    }

    if (node.data.wikilinks && node.data.wikilinks.length > 0) {
      entry.wikilinks = node.data.wikilinks.map(w => ({
        resolvedSlug: w.resolvedSlug,
        resolvedHref: w.resolvedHref,
        missing: Boolean(w.missing),
      }))
    }

    if (Object.keys(entry).length > 0) {
      meta[id] = entry
    }
  }

  return meta
}

export async function processCanvasFile(
  canvasContent: string,
  ctx: BuildCtx,
  currentSlug: FullSlug,
): Promise<JcastCanvas> {
  const jcast = parseJsonCanvas(canvasContent)
  const catalog = referenceCatalogFor(ctx)

  // resolve file nodes
  visitJcast(jcast, 'canvasNode', node => {
    if (!isJcastCanvasNode(node)) return

    const canvasNode = node.data?.canvas
    if (!canvasNode) return

    if (canvasNode.type === 'file') {
      let filePath = canvasNode.file
      if (!filePath || !node.data) return
      let subpath = canvasNode.subpath
      let resolution = catalog.resolve(filePath, currentSlug)
      if (resolution.kind === 'missing' && !subpath && filePath.includes('#')) {
        const separator = filePath.indexOf('#')
        subpath = filePath.slice(separator)
        filePath = filePath.slice(0, separator)
        resolution = catalog.resolve(filePath, currentSlug)
      }
      const request = referenceFromParts({ target: filePath, subpath, embed: true })
      node.data.fileExists = resolution.kind === 'found'
      if (resolution.kind !== 'found') {
        if (resolution.kind !== 'missing') return
        const route = missingReferenceRoute(
          request.parts.target,
          currentSlug,
          request.parts.subpath?.replace(/^#/, '').split('#'),
        )
        if (!route) return
        node.data.resolvedPath = route.slug
        node.data.resolved = {
          slug: route.slug,
          href: `${route.slug}${route.anchor ?? ''}`,
          displayName: filePath.split('/').at(-1)?.replace(/\.md$/, '') ?? filePath,
        }
        return
      }
      const interpreted = interpretReference(resolution.entry, {
        fragments: request.parts.subpath?.replace(/^#/, '').split('#'),
      })
      const route = referenceRoute(resolution.entry, interpreted.selector)
      node.data.resolvedPath = route.slug
      node.data.resolved = {
        slug: route.slug,
        href: `${route.slug}${route.anchor ?? ''}`,
        displayName: resolution.entry.path.split('/').at(-1)?.replace(/\.md$/, '') ?? filePath,
      }
      return
    }

    if (canvasNode.type === 'text') {
      const rawText = canvasNode.text ?? ''
      if (!rawText) {
        return
      }

      const references = extractLiteralReferences(rawText)
      if (references.length === 0) {
        return
      }

      const resolvedLinks: JcastResolvedWikilink[] = references.map(reference => {
        const request = referenceRequest(reference, currentSlug)
        const resolution = catalog.resolve(request.target, currentSlug, { implicitAlias: true })
        if (resolution.kind !== 'found') return { reference, missing: true }
        const interpreted = interpretReference(resolution.entry, {
          fragments: request.fragments,
          settings: request.settings,
        })
        const resolved = referenceRoute(resolution.entry, interpreted.selector)
        const baseHref = resolveRelative(currentSlug, resolved.slug)
        const href = resolved.anchor ? `${baseHref}${resolved.anchor}` : baseHref
        return {
          reference,
          implicitLabel: resolution.implicitLabel,
          resolvedSlug: resolved.slug,
          resolvedHref: href,
          missing: false,
        }
      })

      let idx = 0
      let rewrittenText = ''
      let lastIndex = 0

      for (const reference of references) {
        const { start, end } = reference.span
        if (start > lastIndex) {
          rewrittenText += rawText.slice(lastIndex, start)
        }

        const info = resolvedLinks[idx++]
        const value = rawText.slice(start, end)
        if (!info) {
          rewrittenText += value
          lastIndex = end
          continue
        }

        const display = (
          reference.label?.raw ??
          info.implicitLabel ??
          referenceFields(reference).target
        ).trim()
        if (!info.resolvedHref) {
          rewrittenText += display.length > 0 ? display : reference.raw
          lastIndex = end
          continue
        }

        const label = display.length > 0 ? display : info.resolvedHref
        rewrittenText += `[${label}](${info.resolvedHref})`
        lastIndex = end
      }

      if (lastIndex < rawText.length) {
        rewrittenText += rawText.slice(lastIndex)
      }

      if (node.data) {
        node.data.wikilinks = resolvedLinks
        node.data.resolvedText = rewrittenText
      }
    }
  })

  return jcast
}

declare module 'vfile' {
  interface DataMap {
    canvas?: JcastCanvas
    jsonCanvas?: boolean
    canvasContent?: string
  }
}
