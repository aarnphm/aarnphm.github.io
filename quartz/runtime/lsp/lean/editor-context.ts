import { Facet } from '@codemirror/state'
import type { LspConfig } from '../bridge'

export const leanLspConfigFacet = Facet.define<LspConfig, LspConfig | undefined>({
  combine: values => values.at(-1),
})

export function isRemoteBrowser(): boolean {
  if (typeof window === 'undefined' || typeof WebSocket === 'undefined') return false
  const host = window.location?.hostname
  return Boolean(host && host !== 'localhost' && host !== '127.0.0.1')
}
