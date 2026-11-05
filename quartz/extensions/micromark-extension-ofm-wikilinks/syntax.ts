import type { Extension } from 'micromark-util-types'
import { tokenizeReference } from '../../util/references/lexer'
import './types'

export function wikilink(): Extension {
  return {
    text: {
      33: { name: 'wikilink', tokenize: tokenizeReference },
      91: { name: 'wikilink', tokenize: tokenizeReference },
    },
  }
}
