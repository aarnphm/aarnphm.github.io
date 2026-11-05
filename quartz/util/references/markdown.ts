import type { Nodes, Root } from 'mdast'
import type { VFile } from 'vfile'
import { randomUUID } from 'node:crypto'
import type { BuildCtx } from '../ctx'
import type { ReferenceDependency, ReferenceRequestRecord } from './dependencies'
import { isWikilink } from '../../extensions/micromark-extension-ofm-wikilinks/node'
import { referenceFileKind } from './catalog'
import { referenceCatalogFor, referenceSourcePath } from './context'
import { referenceRequest } from './request'
import { referenceRoute } from './route'
import { interpretReference } from './target'

function decodeFragment(fragment: string): string {
  try {
    return decodeURIComponent(fragment)
  } catch {
    return fragment
  }
}

function resolveRequest(ctx: BuildCtx, request: ReferenceRequestRecord) {
  const resolution = referenceCatalogFor(ctx).resolve(request.target, request.origin, {
    implicitAlias: request.implicitAlias,
    decodeUri: request.decodeUri,
  })
  if (resolution.kind !== 'found')
    return {
      resolution,
      interpretation: interpretReference(
        { path: request.target, kind: referenceFileKind(request.target) },
        request,
      ),
    }
  const interpretation = interpretReference(resolution.entry, {
    ...request,
    fragments: request.decodeUri ? request.fragments.map(decodeFragment) : request.fragments,
  })
  return {
    resolution,
    interpretation,
    route: referenceRoute(resolution.entry, interpretation.selector),
  }
}

function signature(ctx: BuildCtx, requests: readonly ReferenceRequestRecord[]): string {
  return JSON.stringify(
    requests.map(request => {
      const result = resolveRequest(ctx, request)
      return result.resolution.kind === 'found'
        ? [
            result.resolution.kind,
            result.resolution.entry.path,
            result.route,
            result.interpretation?.selector,
            result.interpretation?.settings,
            result.resolution.implicitLabel,
            result.interpretation?.diagnostics,
          ]
        : [
            result.resolution.kind,
            result.resolution.lookupKeys,
            result.resolution.kind === 'ambiguous'
              ? result.resolution.candidates.map(entry => entry.path)
              : undefined,
          ]
    }),
  )
}

export function referenceResolutionSignature(ctx: BuildCtx, file: VFile): string {
  return signature(ctx, file.data.referenceDependencies ?? [])
}

function resolveDependency(ctx: BuildCtx, request: ReferenceRequestRecord): ReferenceDependency {
  const result = resolveRequest(ctx, request)
  const selector = result.interpretation?.selector
  const members =
    request.embed && selector?.kind === 'note-stack'
      ? selector.slugs.map(target =>
          resolveDependency(ctx, {
            target: `/${target}`,
            origin: request.origin,
            fragments: [],
            embed: true,
            decodeUri: false,
            implicitAlias: false,
          }),
        )
      : undefined
  const record: ReferenceDependency = {
    ...request,
    status: result.resolution.kind,
    lookupKeys: result.resolution.lookupKeys,
    ...(result.resolution.kind === 'found'
      ? {
          resolvedPath: result.resolution.entry.path,
          resolvedSlug: result.resolution.entry.slug,
          route: result.route,
          implicitLabel: result.resolution.implicitLabel,
        }
      : {}),
    selector: result.interpretation?.selector,
    resolvedSettings: result.interpretation?.settings,
    diagnostics: result.interpretation?.diagnostics ?? [],
    ...(members ? { members } : {}),
  }
  return record
}

export function recordReferenceDependency(
  ctx: BuildCtx,
  file: VFile,
  request: ReferenceRequestRecord,
): ReferenceDependency {
  const record = resolveDependency(ctx, request)
  // The parent stays last because its position is the Markdown-to-HAST transport identity.
  ;(file.data.referenceDependencies ??= []).push(...(record.members ?? []), record)
  return record
}

function requestForNode(node: Nodes, origin: string): ReferenceRequestRecord | undefined {
  if (isWikilink(node) && node.data?.reference) {
    const reference = node.data.reference
    return referenceRequest(reference, origin, node.data.referenceFields)
  }
  if (node.type === 'code') return node.data?.referenceRequest
  if (node.type !== 'link' && node.type !== 'image') return undefined
  const [target, ...fragments] = node.url.split('#')
  return {
    target,
    origin,
    fragments,
    embed: node.type === 'image',
    decodeUri: true,
    implicitAlias: false,
    ...(node.position?.start.offset !== undefined && node.position.end.offset !== undefined
      ? { authoredSpan: { start: node.position.start.offset, end: node.position.end.offset } }
      : {}),
  }
}

export function visitReferenceNodes(tree: Root, callback: (node: Nodes) => void): void {
  function walk(node: Nodes): void {
    callback(node)
    const labels =
      node.type === 'sidenote'
        ? node.data?.sidenoteParsed?.labelNodes
        : node.type === 'sidenoteReference' || node.type === 'sidenoteDefinition'
          ? node.labelNodes
          : undefined
    if (labels) for (const label of labels) walk(label)
    if ('children' in node) for (const child of node.children) walk(child)
  }
  walk(tree)
}

export function documentReferenceRequests(
  ctx: BuildCtx,
  tree: Root,
  file: VFile,
): ReferenceRequestRecord[] {
  const requests: ReferenceRequestRecord[] = []
  const origin = referenceSourcePath(file.data)
  visitReferenceNodes(tree, node => {
    const request = requestForNode(node, origin)
    if (request) requests.push(...(resolveDependency(ctx, request).members ?? []), request)
  })
  return requests
}

export function resolveDocumentReferences(ctx: BuildCtx, tree: Root, file: VFile): void {
  const origin = referenceSourcePath(file.data)
  file.data.referenceDependencies = []
  file.data.referenceTransportPrefix = `${randomUUID()}:`
  visitReferenceNodes(tree, node => {
    const request = requestForNode(node, origin)
    if (!request) return
    const record = recordReferenceDependency(ctx, file, request)
    const referenceId =
      file.data.referenceTransportPrefix +
      String((file.data.referenceDependencies?.length ?? 1) - 1)
    node.data ??= {}
    node.data.resolvedReference = record
    node.data.referenceId = referenceId
    if (isWikilink(node)) return
    node.data.hProperties = { ...node.data.hProperties, 'data-quartz-reference': referenceId }
    if (
      (node.type === 'link' || node.type === 'image') &&
      record.route &&
      record.status === 'found'
    ) {
      node.url = `${request.target || node.type === 'image' ? record.route.slug : ''}${record.route.anchor ?? ''}`
    }
  })
}
