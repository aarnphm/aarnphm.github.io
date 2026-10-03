export type CharacterMotion = 'f' | 'F' | 't' | 'T'

export interface LeapTarget {
  from: number
  to: number
  cursor: number
}

export interface SyntaxTarget {
  from: number
  to: number
  name: string
}

export interface LeapOptions {
  vimMotions: boolean
  readingMotions: boolean
  showLabels: boolean
}
