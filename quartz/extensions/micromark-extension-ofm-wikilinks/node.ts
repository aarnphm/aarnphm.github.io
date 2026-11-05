import type { Data } from 'mdast'
import type { Literal } from 'unist'
import type { ReferenceSyntax } from '../../util/references/model'
import { isRecord } from '../../util/type-guards'

export interface Wikilink extends Literal {
  type: 'wikilink'
  value: string
  data?: WikilinkNodeData
}

export interface WikilinkNodeData extends Data {
  reference?: ReferenceSyntax
}

export function isWikilink(node: unknown): node is Wikilink {
  return (
    isRecord(node) &&
    node.type === 'wikilink' &&
    typeof node.value === 'string' &&
    isRecord(node.data) &&
    isRecord(node.data.reference) &&
    typeof node.data.reference.raw === 'string' &&
    typeof node.data.reference.embed === 'boolean'
  )
}

declare module 'mdast' {
  interface PhrasingContentMap {
    wikilink: Wikilink
  }
  interface RootContentMap {
    wikilink: Wikilink
  }
}
