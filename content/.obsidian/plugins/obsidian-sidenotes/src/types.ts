export interface SidenoteProperties {
  [key: string]: string | string[]
}

export interface ParsedSidenote {
  reference?: boolean
  raw: string
  properties?: SidenoteProperties
  label?: string
  content: string
}

export interface SidenoteMatch {
  from: number
  to: number
  contentFrom: number
  data: ParsedSidenote
}

export interface SidenoteDefinition {
  from: number
  to: number
  lineStart: number
  lineEnd: number
  label: string
  content: string
}
