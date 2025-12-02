export interface SidenoteProperties {
  [key: string]: string | string[]
}

export interface ParsedSidenote {
  raw: string
  properties?: SidenoteProperties
  label?: string
  content: string
}

export type TextSegment = {
  type: "text"
  value: string
}

export type SidenoteSegment = {
  type: "sidenote"
  data: ParsedSidenote
}

export type Segment = TextSegment | SidenoteSegment
