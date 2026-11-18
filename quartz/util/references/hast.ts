import type { Element, Root } from 'hast'
import type { VFile } from 'vfile'
import { visit } from 'unist-util-visit'
import type { FullSlug } from '../path'
import { referenceHref } from './route'

const transportProperties = ['data-quartz-reference', 'dataQuartzReference']

/** Raw HTML parsing replaces node objects; document-local identities retain their typed records. */
export function attachHastReferences(tree: Root, file: VFile): void {
  const references = file.data.referenceDependencies
  const prefix = file.data.referenceTransportPrefix
  if (!references?.length || !prefix) return
  visit(tree, 'element', node => {
    const id = node.properties['data-quartz-reference'] ?? node.properties.dataQuartzReference
    if (typeof id !== 'string' || !id.startsWith(prefix)) return
    const index = id.slice(prefix.length)
    if (!/^\d+$/.test(index)) return
    const request = references[Number(index)]
    if (!request) return
    node.data ??= { position: {} }
    node.data.referenceRequest = request
  })
}

export function clearReferenceTransport(node: Element): void {
  for (const property of transportProperties) delete node.properties[property]
}

export function stripReferenceTransport(tree: Root, file: VFile): void {
  visit(tree, 'element', clearReferenceTransport)
  for (const block of Object.values(file.data.blocks ?? {}))
    visit(block, 'element', clearReferenceTransport)
  delete file.data.referenceTransportPrefix
}

/** Occurrence copies project known destinations from identity after generic relative-URL rebasing. */
export function rebaseReferenceElement(node: Element, from: FullSlug): void {
  const request = node.data?.referenceRequest
  if (request?.status !== 'found' || !request.route) return
  if (typeof node.properties.href === 'string' && (request.target || request.embed))
    node.properties.href = referenceHref(from, request.route)
  for (const property of ['src', 'data-pdf-src', 'dataPdfSrc']) {
    const value = node.properties[property]
    if (typeof value !== 'string' || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(value)) continue
    const hash = value.indexOf('#')
    node.properties[property] = referenceHref(from, {
      slug: request.route.slug,
      ...(hash >= 0 ? { anchor: value.slice(hash) } : {}),
    })
  }
}
