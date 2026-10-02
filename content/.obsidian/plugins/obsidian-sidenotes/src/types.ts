export interface SidenoteProperties {
  [key: string]: string | string[]
}

export interface ParsedSidenote {
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
