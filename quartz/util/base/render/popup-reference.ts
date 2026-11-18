import type { PopupReference } from '../../../components/scripts/base-map-data'
import type { FullSlug } from '../../path'
import { labelToText, parseReferenceLabel } from '../../references/label'
import {
  parseLiteralReference,
  resolveLiteralReference,
  referenceLabelSource,
  type LiteralReferenceResolutionOptions,
} from '../../references/literal'

export function projectPopupValue(
  value: unknown,
  currentSlug: FullSlug,
  options: LiteralReferenceResolutionOptions,
): unknown {
  if (Array.isArray(value)) return value.map(item => projectPopupValue(item, currentSlug, options))
  if (typeof value !== 'string') return value
  const link = parseLiteralReference(value)
  if (!link) return value
  const resolved = resolveLiteralReference(link, currentSlug, options)
  if (!resolved) return value
  const reference: PopupReference = {
    kind: 'reference',
    href: `/${resolved.slug}${resolved.anchor ?? ''}`,
    label: labelToText(
      parseReferenceLabel(referenceLabelSource(link, resolved.slug, resolved.implicitLabel)),
    ),
  }
  return reference
}
