import type { Code, State } from 'micromark-util-types'

export type ReferenceToken =
  | 'wikilink'
  | 'wikilinkEmbedMarker'
  | 'wikilinkOpenMarker'
  | 'wikilinkCloseMarker'
  | 'wikilinkTarget'
  | 'wikilinkTargetChunk'
  | 'wikilinkAnchorMarker'
  | 'wikilinkAnchor'
  | 'wikilinkAnchorChunk'
  | 'wikilinkMetadataMarker'
  | 'wikilinkMetadata'
  | 'wikilinkMetadataChunk'
  | 'wikilinkMetadataCloseMarker'
  | 'wikilinkAliasMarker'
  | 'wikilinkAlias'
  | 'wikilinkAliasChunk'

export interface ReferenceEffects {
  enter(type: ReferenceToken, fields?: { contentType: 'string' }): unknown
  exit(type: ReferenceToken): unknown
  consume(code: Code): void
}

const codes = {
  exclamationMark: 33,
  numberSign: 35,
  leftCurlyBrace: 123,
  rightCurlyBrace: 125,
  backslash: 92,
  leftSquareBracket: 91,
  rightSquareBracket: 93,
  verticalBar: 124,
}

export function tokenizeReference(effects: ReferenceEffects, ok: State, nok: State): State {
  let previousWasBackslash = false

  return start

  function start(code: Code): State | undefined {
    if (code === codes.exclamationMark) {
      effects.enter('wikilink')
      effects.enter('wikilinkEmbedMarker')
      effects.consume(code)
      effects.exit('wikilinkEmbedMarker')
      return openFirst
    }

    if (code === codes.leftSquareBracket) {
      effects.enter('wikilink')
      return openFirst(code)
    }

    return nok(code)
  }

  function openFirst(code: Code): State | undefined {
    if (code !== codes.leftSquareBracket) {
      return nok(code)
    }

    effects.enter('wikilinkOpenMarker')
    effects.consume(code)
    return openSecond
  }

  function openSecond(code: Code): State | undefined {
    if (code !== codes.leftSquareBracket) {
      return nok(code)
    }

    effects.consume(code)
    effects.exit('wikilinkOpenMarker')
    return targetStart
  }

  function targetStart(code: Code): State | undefined {
    if (code === codes.numberSign) {
      return anchorMarker(code)
    }
    if (code === codes.verticalBar) {
      return aliasMarker(code)
    }
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter('wikilinkTarget')
      effects.enter('wikilinkTargetChunk', { contentType: 'string' })
      return targetInside(code)
    }

    return nok(code)
  }

  function targetInside(code: Code): State | undefined {
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return targetInside
    }
    if (code === codes.numberSign && !previousWasBackslash) {
      effects.exit('wikilinkTargetChunk')
      effects.exit('wikilinkTarget')
      previousWasBackslash = false
      return anchorMarker(code)
    }
    if (code === codes.verticalBar) {
      effects.exit('wikilinkTargetChunk')
      effects.exit('wikilinkTarget')
      previousWasBackslash = false
      return aliasMarker(code)
    }
    if (code === codes.rightSquareBracket && !previousWasBackslash) {
      effects.exit('wikilinkTargetChunk')
      effects.exit('wikilinkTarget')
      previousWasBackslash = false
      return closeFirst(code)
    }
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }
    effects.consume(code)
    previousWasBackslash = false
    return targetInside
  }

  function anchorMarker(code: Code): State | undefined {
    if (code !== codes.numberSign) {
      return nok(code)
    }

    effects.enter('wikilinkAnchorMarker')
    effects.consume(code)
    return anchorMarkerDecide
  }

  function anchorMarkerDecide(code: Code): State | undefined {
    if (code === codes.leftCurlyBrace) {
      effects.exit('wikilinkAnchorMarker')
      effects.enter('wikilinkMetadataMarker')
      effects.consume(code) // consume {
      effects.exit('wikilinkMetadataMarker')
      return metadataStart
    }
    effects.exit('wikilinkAnchorMarker')
    return anchorStart(code)
  }

  function anchorStart(code: Code): State | undefined {
    if (code === codes.numberSign) return anchorMarker(code)
    if (code === codes.verticalBar) {
      return aliasMarker(code)
    }
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter('wikilinkAnchor')
      effects.enter('wikilinkAnchorChunk', { contentType: 'string' })
      return anchorInside(code)
    }

    return nok(code)
  }

  function anchorInside(code: Code): State | undefined {
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return anchorInside
    }
    if (code === codes.numberSign && !previousWasBackslash) {
      effects.exit('wikilinkAnchorChunk')
      effects.exit('wikilinkAnchor')
      previousWasBackslash = false
      return anchorOrMetadata(code)
    }
    if (code === codes.verticalBar) {
      effects.exit('wikilinkAnchorChunk')
      effects.exit('wikilinkAnchor')
      previousWasBackslash = false
      return aliasMarker(code)
    }
    if (code === codes.rightSquareBracket && !previousWasBackslash) {
      effects.exit('wikilinkAnchorChunk')
      effects.exit('wikilinkAnchor')
      previousWasBackslash = false
      return closeFirst(code)
    }
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }
    effects.consume(code)
    previousWasBackslash = false
    return anchorInside
  }

  function anchorOrMetadata(code: Code): State | undefined {
    if (code !== codes.numberSign) {
      return nok(code)
    }

    effects.enter('wikilinkAnchorMarker')
    effects.consume(code)
    return anchorOrMetadataDecide
  }

  function anchorOrMetadataDecide(code: Code): State | undefined {
    effects.exit('wikilinkAnchorMarker')
    if (code === codes.leftCurlyBrace) {
      effects.enter('wikilinkMetadataMarker')
      effects.consume(code) // consume {
      effects.exit('wikilinkMetadataMarker')
      return metadataStart
    }

    if (code === codes.numberSign) {
      return anchorOrMetadata(code)
    }
    return anchorStart(code)
  }

  function metadataStart(code: Code): State | undefined {
    if (code === codes.rightCurlyBrace) {
      return metadataEnd(code)
    }
    if (code === codes.verticalBar) {
      return aliasMarker(code)
    }
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter('wikilinkMetadata')
      effects.enter('wikilinkMetadataChunk', { contentType: 'string' })
      return metadataInside(code, 1) // start with depth 1 (already consumed opening {)
    }

    return nok(code)
  }

  function metadataInside(code: Code, braceDepth: number): State | undefined {
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return (nextCode: Code) => metadataInside(nextCode, braceDepth)
    }
    if (code === codes.leftCurlyBrace && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = false
      return (nextCode: Code) => metadataInside(nextCode, braceDepth + 1)
    }
    if (code === codes.rightCurlyBrace && !previousWasBackslash) {
      if (braceDepth === 1) {
        effects.exit('wikilinkMetadataChunk')
        effects.exit('wikilinkMetadata')
        previousWasBackslash = false
        return metadataEnd(code)
      }
      effects.consume(code)
      previousWasBackslash = false
      return (nextCode: Code) => metadataInside(nextCode, braceDepth - 1)
    }
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }
    effects.consume(code)
    previousWasBackslash = false
    return (nextCode: Code) => metadataInside(nextCode, braceDepth)
  }

  function metadataEnd(code: Code): State | undefined {
    if (code !== codes.rightCurlyBrace) {
      return nok(code)
    }

    effects.enter('wikilinkMetadataCloseMarker')
    effects.consume(code)
    effects.exit('wikilinkMetadataCloseMarker')
    return (nextCode: Code) => {
      if (nextCode === codes.backslash) {
        effects.consume(nextCode)
        return (escapedCode: Code) =>
          escapedCode === codes.verticalBar ? aliasMarker(escapedCode) : nok(escapedCode)
      }
      if (nextCode === codes.verticalBar) {
        return aliasMarker(nextCode)
      }

      if (nextCode === codes.rightSquareBracket) {
        return closeFirst(nextCode)
      }

      return nok(nextCode)
    }
  }

  function aliasMarker(code: Code): State | undefined {
    if (code !== codes.verticalBar) {
      return nok(code)
    }

    effects.enter('wikilinkAliasMarker')
    effects.consume(code)
    effects.exit('wikilinkAliasMarker')
    return aliasStart
  }

  function aliasStart(code: Code): State | undefined {
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter('wikilinkAlias')
      effects.enter('wikilinkAliasChunk', { contentType: 'string' })
      return aliasInside(code)
    }

    return nok(code)
  }

  function aliasInside(code: Code): State | undefined {
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return aliasInside
    }
    if (code === codes.rightSquareBracket && !previousWasBackslash) {
      effects.exit('wikilinkAliasChunk')
      effects.exit('wikilinkAlias')
      previousWasBackslash = false
      return closeFirst(code)
    }
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }
    effects.consume(code)
    previousWasBackslash = false
    return aliasInside
  }

  function closeFirst(code: Code): State | undefined {
    if (code !== codes.rightSquareBracket) {
      return nok(code)
    }

    effects.enter('wikilinkCloseMarker')
    effects.consume(code)
    return closeSecond
  }

  function closeSecond(code: Code): State | undefined {
    if (code !== codes.rightSquareBracket) {
      return nok(code)
    }

    effects.consume(code)
    effects.exit('wikilinkCloseMarker')
    effects.exit('wikilink')
    return ok
  }
}
