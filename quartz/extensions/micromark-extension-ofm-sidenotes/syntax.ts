import type { Extension, Tokenizer, State, Code } from "micromark-util-types"

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
}

const KEYWORD = "sidenotes"

export function sidenote(): Extension {
  const tokenize: Tokenizer = function (effects, ok, nok) {
    let markerSize = 0
    let keywordIndex = 0
    let labelBalance = 0

    return start

    function start(code: Code): State | undefined {
      if (code !== codes.leftCurlyBrace) return nok(code)

      effects.enter("sidenote")
      effects.enter("sidenoteMarker")
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
          effects.exit("sidenoteMarker")
          return keyword
        }
        return openingBrace
      }
      return nok(code)
    }

    function keyword(code: Code): State | undefined {
      if (keywordIndex === 0) {
        effects.enter("sidenoteKeyword")
      }

      if (keywordIndex < KEYWORD.length) {
        if (code === KEYWORD.charCodeAt(keywordIndex)) {
          effects.consume(code)
          keywordIndex++
          return keyword
        }
        return nok(code)
      }

      effects.exit("sidenoteKeyword")
      return afterKeyword(code)
    }

    function afterKeyword(code: Code): State | undefined {
      if (code === codes.lessThan) {
        return propertiesStart(code)
      }
      if (code === codes.leftSquareBracket) {
        return labelStart(code)
      }
      if (code === codes.colon) {
        return colonMarker(code)
      }
      return nok(code)
    }

    function propertiesStart(code: Code): State | undefined {
      effects.enter("sidenoteProperties")
      effects.enter("sidenotePropertiesMarker")
      effects.consume(code)
      effects.exit("sidenotePropertiesMarker")
      effects.enter("sidenotePropertiesChunk")
      return propertiesInside
    }

    function propertiesInside(code: Code): State | undefined {
      if (code === codes.greaterThan) {
        effects.exit("sidenotePropertiesChunk")
        effects.enter("sidenotePropertiesMarker")
        effects.consume(code)
        effects.exit("sidenotePropertiesMarker")
        effects.exit("sidenoteProperties")
        return afterProperties
      }

      if (code === codes.eof || code === null) {
        return nok(code)
      }

      effects.consume(code)
      return propertiesInside
    }

    function afterProperties(code: Code): State | undefined {
      if (code === codes.leftSquareBracket) {
        return labelStart(code)
      }
      if (code === codes.colon) {
        return colonMarker(code)
      }
      return nok(code)
    }

    function labelStart(code: Code): State | undefined {
      effects.enter("sidenoteLabel")
      effects.enter("sidenoteLabelMarker")
      effects.consume(code)
      effects.exit("sidenoteLabelMarker")
      effects.enter("sidenoteLabelChunk")
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
        effects.exit("sidenoteLabelChunk")
        effects.enter("sidenoteLabelMarker")
        effects.consume(code)
        effects.exit("sidenoteLabelMarker")
        effects.exit("sidenoteLabel")
        return afterLabel
      }

      if (code === codes.eof || code === null) {
        return nok(code)
      }

      effects.consume(code)
      return labelInside
    }

    function afterLabel(code: Code): State | undefined {
      if (code === codes.colon) {
        return colonMarker(code)
      }
      if (code === codes.lessThan) {
        return propertiesStart(code)
      }
      if (code === codes.rightCurlyBrace) {
        return closingBraceFirst(code)
      }
      return nok(code)
    }

    function colonMarker(code: Code): State | undefined {
      effects.enter("sidenoteColonMarker")
      effects.consume(code)
      effects.exit("sidenoteColonMarker")
      return contentStart
    }

    function contentStart(_: Code): State | undefined {
      effects.enter("sidenoteContent")
      effects.enter("sidenoteContentChunk")
      return contentInside
    }

    function contentInside(code: Code): State | undefined {
      if (code === codes.rightCurlyBrace) {
        effects.exit("sidenoteContentChunk")
        effects.exit("sidenoteContent")
        return closingBraceFirst(code)
      }

      if (code === codes.eof || code === null) {
        return nok(code)
      }

      effects.consume(code)
      return contentInside
    }

    function closingBraceFirst(code: Code): State | undefined {
      effects.enter("sidenoteMarker")
      effects.consume(code)
      return closingBraceSecond
    }

    function closingBraceSecond(code: Code): State | undefined {
      if (code === codes.rightCurlyBrace) {
        effects.consume(code)
        effects.exit("sidenoteMarker")
        effects.exit("sidenote")
        return ok
      }
      return nok(code)
    }
  }

  return { text: { [codes.leftCurlyBrace]: { tokenize, resolveAll: resolveAllSidenote } } }
}

function resolveAllSidenote(events: any[]) {
  for (let i = 0; i < events.length; i++) {
    const event = events[i]
    if (event[0] === "enter" && event[1].type === "sidenote") {
      let hasContent = false
      let j = i + 1
      while (j < events.length) {
        if (events[j][0] === "exit" && events[j][1] === event[1]) break
        if (events[j][1].type === "sidenoteContent") {
          hasContent = true
          break
        }
        j++
      }

      if (!hasContent) {
        event[1].type = "sidenoteReference"
        let k = i + 1
        while (k < events.length) {
          if (events[k][0] === "exit" && events[k][1] === event[1]) break
          const type = events[k][1].type as string
          if (type.startsWith("sidenote") && !type.startsWith("sidenoteReference")) {
            events[k][1].type = type.replace("sidenote", "sidenoteReference")
          }
          k++
        }
      }
    }
  }
  return events
}
