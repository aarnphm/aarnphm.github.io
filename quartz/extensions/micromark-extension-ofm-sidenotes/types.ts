import type { Parent, PhrasingContent, BlockContent } from 'mdast'
import type { Point } from 'unist'

declare module 'micromark-util-types' {
  interface TokenTypeMap {
    sidenote: 'sidenote'
    sidenoteMarker: 'sidenoteMarker'
    sidenoteKeyword: 'sidenoteKeyword'
    sidenotePropertiesMarker: 'sidenotePropertiesMarker'
    sidenoteProperties: 'sidenoteProperties'
    sidenotePropertiesChunk: 'sidenotePropertiesChunk'
    sidenoteLabelMarker: 'sidenoteLabelMarker'
    sidenoteLabel: 'sidenoteLabel'
    sidenoteLabelChunk: 'sidenoteLabelChunk'
    sidenoteColonMarker: 'sidenoteColonMarker'
    sidenoteContent: 'sidenoteContent'
    sidenoteContentChunk: 'sidenoteContentChunk'
    // Reference types
    sidenoteReference: 'sidenoteReference'
    sidenoteReferenceMarker: 'sidenoteReferenceMarker'
    sidenoteReferenceKeyword: 'sidenoteReferenceKeyword'
    sidenoteReferenceLabelMarker: 'sidenoteReferenceLabelMarker'
    sidenoteReferenceLabel: 'sidenoteReferenceLabel'
    sidenoteReferenceLabelChunk: 'sidenoteReferenceLabelChunk'
    // Definition types
    sidenoteDefinition: 'sidenoteDefinition'
    sidenoteDefinitionMarker: 'sidenoteDefinitionMarker'
    sidenoteDefinitionLabel: 'sidenoteDefinitionLabel'
    sidenoteDefinitionLabelMarker: 'sidenoteDefinitionLabelMarker'
    sidenoteDefinitionLabelChunk: 'sidenoteDefinitionLabelChunk'
    sidenoteDefinitionWhitespace: 'sidenoteDefinitionWhitespace'
  }
}

export type { BlockContent }

export interface SidenoteData {
  raw: string
  properties?: Record<string, string | string[]>
  label?: string
  labelNodes?: PhrasingContent[]
  content: string
}

export interface Sidenote extends Parent {
  type: 'sidenote'
  value: string
  children: PhrasingContent[]
  position?: { start: Point; end: Point }
  properties?: Record<string, unknown>
}

export interface SidenoteReference extends Parent {
  type: 'sidenoteReference'
  label: string
  labelNodes?: PhrasingContent[]
  children: PhrasingContent[]
  properties?: Record<string, unknown>
}

export interface SidenoteDefinition extends Parent {
  type: 'sidenoteDefinition'
  label: string
  labelNodes?: PhrasingContent[]
  children: BlockContent[]
}

declare module 'mdast' {
  interface Data {
    sidenoteParsed?: SidenoteData
    sidenoteId?: number
    baseId?: string
    forceInline?: boolean
    allowLeft?: boolean
    allowRight?: boolean
    internal?: string | string[]
    label?: string
  }

  interface PhrasingContentMap {
    sidenote: Sidenote
    sidenoteReference: SidenoteReference
  }

  interface BlockContentMap {
    sidenoteDefinition: SidenoteDefinition
  }

  interface RootContentMap {
    sidenote: Sidenote
    sidenoteReference: SidenoteReference
    sidenoteDefinition: SidenoteDefinition
  }
}
