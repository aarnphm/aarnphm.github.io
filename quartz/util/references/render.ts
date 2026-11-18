import type { Element } from 'hast'
import type { Handler } from 'mdast-util-to-hast'
import { h } from 'hastscript'
import type { ReferenceDependency } from './dependencies'
import type { ReferenceSyntax } from './model'
import type { ReferenceFields } from './request'
import { isWikilink } from '../../extensions/micromark-extension-ofm-wikilinks/node'
import { buildExternalEmbed } from '../embed'
import { buildYouTubeEmbed } from '../youtube'
import { referenceFileKind } from './catalog'
import { labelToHast, parseReferenceLabel } from './label'
import { imageReferenceElement, pdfReferenceElement } from './render-media'
import { referenceRequest } from './request'
import { missingReferenceRoute } from './route'
import { interpretReference } from './target'

/** Standalone syntax rendering projects a destination without claiming catalog membership. */
function unresolvedReference(
  reference: ReferenceSyntax,
  fields?: ReferenceFields,
): ReferenceDependency {
  const request = referenceRequest(reference, 'index.md', fields)
  const route = missingReferenceRoute(request.target, request.origin, request.fragments)
  const interpretation = interpretReference(
    { path: request.target, kind: referenceFileKind(request.target) },
    request,
  )
  return {
    ...request,
    status: /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(request.target) ? 'external' : 'missing',
    lookupKeys: [],
    route,
    selector: interpretation.selector,
    resolvedSettings: interpretation.settings,
    diagnostics: interpretation.diagnostics,
  }
}

export function referenceElement(
  record: ReferenceDependency,
  reference?: ReferenceSyntax,
): Element {
  const target = record.target.trim()
  const kind = referenceFileKind(record.resolvedPath ?? target)
  const external = /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(target)
  const route = record.route ?? missingReferenceRoute(target, record.origin, record.fragments)
  const url =
    external || (target.startsWith('/') && kind !== 'flashcard')
      ? target
      : target || record.embed
        ? `${target.startsWith('/') ? '/' : ''}${route?.slug ?? target}`
        : ''
  const anchor = external
    ? record.fragments.length
      ? `#${record.fragments.join('#')}`
      : ''
    : (route?.anchor ?? '')
  const label = record.label ?? record.implicitLabel
  let element: Element
  if (!record.embed) {
    const defaultLabel = [target, ...record.fragments.map(fragment => fragment.replace(/^\^/, ''))]
      .filter(Boolean)
      .join(' > ')
    const source =
      reference?.label?.raw ??
      label ??
      (record.selector?.kind === 'base' ? record.selector.view : defaultLabel)
    element = h('a', { href: url + anchor }, labelToHast(parseReferenceLabel(source)))
  } else if (buildYouTubeEmbed(target + anchor)) {
    element = h('img', { src: target + anchor })
  } else if (kind === 'image') {
    element = imageReferenceElement(record, url)
  } else if (kind === 'video') {
    element = h('video', { src: url, controls: true, loop: true })
  } else if (kind === 'audio') {
    element = h('audio', {
      src: url,
      controls: true,
      ...(label?.trim() ? { 'data-caption': label.trim() } : {}),
    })
  } else if (kind === 'pdf') {
    element = pdfReferenceElement(record, url)
  } else if (kind === 'canvas') {
    element = h('div', {
      'data-canvas-embed': true,
      'data-canvas-file': record.resolvedPath ?? target,
      'data-canvas-title': label ?? target,
    })
  } else if (/^https?:\/\//i.test(target) && buildExternalEmbed(target + anchor)) {
    const dimensions = /^(\d+%?)(?:x(\d+%?))?$/.exec((label ?? '').split('|').at(-1)?.trim() ?? '')
    element = h('img.external-embed', {
      src: target + anchor,
      ...(dimensions?.[1] ? { width: dimensions[1] } : {}),
      ...(dimensions?.[2] ? { height: dimensions[2] } : {}),
    })
  } else {
    const inner = h('a.transclude-inner', { href: url + anchor }, `Transclude of ${url} ${anchor}`)
    inner.data = { position: {}, referenceRequest: record }
    element = h('blockquote.transclude', [inner])
  }
  element.data = { position: {}, referenceRequest: record }
  return element
}

export const referenceHastHandler: Handler = (state, node) => {
  if (!isWikilink(node) || !node.data?.reference) return
  const record =
    node.data.resolvedReference ??
    unresolvedReference(node.data.reference, node.data.referenceFields)
  const element = referenceElement(record, node.data.reference)
  if (node.data.referenceId !== undefined) {
    element.properties['data-quartz-reference'] = node.data.referenceId
    if (element.tagName === 'blockquote') {
      const inner = element.children[0]
      if (inner?.type === 'element')
        inner.properties['data-quartz-reference'] = node.data.referenceId
    }
  }
  state.patch(node, element)
  if (element.tagName === 'blockquote') {
    const inner = element.children[0]
    if (inner?.type === 'element') state.patch(node, inner)
  }
  return element
}
