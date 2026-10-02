import type { Parent, PhrasingContent } from 'mdast'

export interface SpeechPhrase extends Parent {
  type: 'speechPhrase'
  children: PhrasingContent[]
}

declare module 'micromark-util-types' {
  interface TokenTypeMap {
    speechPhrase: 'speechPhrase'
    speechMarker: 'speechMarker'
  }
}

declare module 'mdast' {
  interface PhrasingContentMap {
    speechPhrase: SpeechPhrase
  }

  interface RootContentMap {
    speechPhrase: SpeechPhrase
  }
}
