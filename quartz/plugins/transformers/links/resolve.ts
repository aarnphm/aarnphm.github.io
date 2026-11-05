import type { Element } from 'hast'
import type { VFile } from 'vfile'
import type { BuildCtx } from '../../../util/ctx'
import type { FullSlug } from '../../../util/path'
import { isFullSlug } from '../../../util/path'
import { referenceSourcePath } from '../../../util/references/context'
import { recordReferenceDependency } from '../../../util/references/markdown'
import { missingReferenceRoute, referenceHref } from '../../../util/references/route'

function decodeUriPart(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function hasCanonicalReference(node: Element): boolean {
  return node.data?.referenceRequest?.status === 'found'
}

/** Selectors are already interpreted by the reference owner; URL rewriting keeps their bytes. */
export function resolveInternalLink(
  ctx: BuildCtx,
  file: VFile,
  node: Element,
  destination: string,
): { href: string; slug: FullSlug } | undefined {
  const fileSlug = file.data.slug
  if (!fileSlug) return undefined
  const hash = destination.indexOf('#')
  const target = hash < 0 ? destination : destination.slice(0, hash)
  const request = node.data?.referenceRequest
  if (request?.route) {
    return {
      href: referenceHref(fileSlug, request.route, !request.target && !request.embed),
      slug: request.route.slug,
    }
  }
  if (target.length === 0) return undefined
  const record = recordReferenceDependency(ctx, file, {
    target,
    origin: referenceSourcePath(file.data),
    fragments: hash < 0 ? [] : destination.slice(hash + 1).split('#'),
    embed: false,
    decodeUri: true,
    implicitAlias: false,
  })
  const route =
    record.route ??
    missingReferenceRoute(
      decodeUriPart(target),
      referenceSourcePath(file.data),
      hash < 0
        ? []
        : destination
            .slice(hash + 1)
            .split('#')
            .map(decodeUriPart),
    )
  if (!route) return undefined
  return { href: referenceHref(fileSlug, route), slug: route.slug }
}

export function canonicalResourceUrl(
  node: Element,
  value: string,
  fileSlug: FullSlug,
): string | undefined {
  if (!hasCanonicalReference(node) || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(value)) return undefined
  const hash = value.indexOf('#')
  const target = hash < 0 ? value : value.slice(0, hash)
  const anchor = hash < 0 ? '' : value.slice(hash)
  if (!isFullSlug(target)) return undefined
  return referenceHref(fileSlug, { slug: target, anchor })
}
