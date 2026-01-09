import type { Extension, Tokenizer, State, Code } from "micromark-util-types"
import { factorySpace } from "micromark-factory-space"

const codes = {
  leftCurlyBrace: 123,
  rightCurlyBrace: 125,
  lessThan: 60,
  greaterThan: 62,
  leftSquareBracket: 91,
  rightSquareBracket: 93,
  colon: 58,
  backslash: 92,
  eof: null,
  space: 32,
  tab: 9,
}

const KEYWORD = "sidenotes"

export function sidenoteDefinition(): Extension {
  const tokenize: Tokenizer = function (effects, ok, nok) {
    let markerSize = 0
    let keywordIndex = 0
    let labelBalance = 0

    return start

    function start(code: Code): State | undefined {
      if (code !== codes.leftCurlyBrace) return nok(code)

      effects.enter("sidenoteDefinition")
      effects.enter("sidenoteDefinitionMarker")
      effects.consume(code)
      markerSize = 1
      labelBalance = 0
      return openingBrace
    }

    function openingBrace(code: Code): State | undefined {
      if (code === codes.leftCurlyBrace) {
        effects.consume(code)
        markerSize++
        if (markerSize === 2) {
          effects.exit("sidenoteDefinitionMarker")
          return keyword
        }
        return openingBrace
      }
      return nok(code)
    }

    function keyword(code: Code): State | undefined {
      if (keywordIndex < KEYWORD.length) {
        if (code === KEYWORD.charCodeAt(keywordIndex)) {
          effects.consume(code)
          keywordIndex++
          return keyword
        }
        return nok(code)
      }
      return afterKeyword(code)
    }

    function afterKeyword(code: Code): State | undefined {
      if (code === codes.leftSquareBracket) {
        return labelStart(code)
      }
      return nok(code)
    }

    function labelStart(code: Code): State | undefined {
      effects.enter("sidenoteDefinitionLabel")
      effects.enter("sidenoteDefinitionLabelMarker")
      effects.consume(code)
      effects.exit("sidenoteDefinitionLabelMarker")
      effects.enter("sidenoteDefinitionLabelChunk")
      labelBalance = 0
      return labelInside
    }

    function labelInside(code: Code): State | undefined {
      if (code === codes.leftSquareBracket) {
        labelBalance++
        effects.consume(code)
        return labelInside
      }

      if (code === codes.rightSquareBracket) {
        if (labelBalance > 0) {
          labelBalance--
          effects.consume(code)
          return labelInside
        }
        effects.exit("sidenoteDefinitionLabelChunk")
        effects.enter("sidenoteDefinitionLabelMarker")
        effects.consume(code)
        effects.exit("sidenoteDefinitionLabelMarker")
        effects.exit("sidenoteDefinitionLabel")
        return afterLabel
      }

      if (code === codes.eof || code === null) {
        return nok(code)
      }

      effects.consume(code)
      return labelInside
    }

    function afterLabel(code: Code): State | undefined {
      if (code === codes.rightCurlyBrace) {
        return closingBraceFirst(code)
      }
      return nok(code)
    }

    function closingBraceFirst(code: Code): State | undefined {
      effects.enter("sidenoteDefinitionMarker")
      effects.consume(code)
      return closingBraceSecond
    }

    function closingBraceSecond(code: Code): State | undefined {
      if (code === codes.rightCurlyBrace) {
        effects.consume(code)
        effects.exit("sidenoteDefinitionMarker")
        return afterDefinitionBlock
      }
      return nok(code)
    }

    function afterDefinitionBlock(code: Code): State | undefined {
      if (code === codes.colon) {
        effects.enter("sidenoteDefinitionMarker")
        effects.consume(code)
        effects.exit("sidenoteDefinitionMarker")
        return factorySpace(effects, atContent, "sidenoteDefinitionWhitespace")
      }
      return nok(code)
    }

    function atContent(code: Code): State | undefined {
      effects.exit("sidenoteDefinition")
      return ok(code)
    }
  }

  return {
    flow: {
      [codes.leftCurlyBrace]: { tokenize },
    },
  }
}
