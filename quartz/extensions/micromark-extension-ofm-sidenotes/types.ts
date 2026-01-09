import type { Node, PhrasingContent } from "mdast"
import type { Point } from "unist"

declare module "micromark-util-types" {
  interface TokenTypeMap {
    sidenote: "sidenote"
    sidenoteMarker: "sidenoteMarker"
    sidenoteKeyword: "sidenoteKeyword"
    sidenotePropertiesMarker: "sidenotePropertiesMarker"
    sidenoteProperties: "sidenoteProperties"
    sidenotePropertiesChunk: "sidenotePropertiesChunk"
    sidenoteLabelMarker: "sidenoteLabelMarker"
    sidenoteLabel: "sidenoteLabel"
    sidenoteLabelChunk: "sidenoteLabelChunk"
    sidenoteColonMarker: "sidenoteColonMarker"
    sidenoteContent: "sidenoteContent"
    sidenoteContentChunk: "sidenoteContentChunk"
  }
}

export interface SidenoteData {
  raw: string
  properties?: Record<string, string | string[]>
  label?: string
  labelNodes?: PhrasingContent[]
  content: string
}

export interface Sidenote extends Node {
  type: "sidenote"
  value: string
  children: PhrasingContent[]
  position?: { start: Point; end: Point }
}

declare module "mdast" {
  interface Data {
    sidenoteParsed?: SidenoteData
  }
}
