import type { Code, Construct, Extension, State, Tokenizer } from 'micromark-util-types'
import { markdownLineEnding } from 'micromark-util-character'
import { codes } from 'micromark-util-symbol'
import './types'

const closingBraces: Construct = {
  partial: true,
  tokenize(effects, ok, nok) {
    return first

    function first(code: Code): State | undefined {
      if (code !== codes.rightCurlyBrace) return nok(code)
      effects.consume(code)
      return second
    }

    function second(code: Code): State | undefined {
      if (code !== codes.rightCurlyBrace) return nok(code)
      effects.consume(code)
      return ok
    }
  },
}

export function speech(): Extension {
  const tokenize: Tokenizer = function (effects, ok, nok) {
    const previous = this.previous
    let prefix = ''
    let hasContent = false
    let codeFence = 0
    let tickRun = 0

    return start

    function start(code: Code): State | undefined {
      if (code !== codes.leftCurlyBrace || previous === codes.leftCurlyBrace) return nok(code)
      effects.enter('speechPhrase')
      effects.enter('speechMarker')
      effects.consume(code)
      return opening
    }

    function opening(code: Code): State | undefined {
      if (code !== codes.leftCurlyBrace) return nok(code)
      effects.consume(code)
      effects.exit('speechMarker')
      effects.enter('chunkText', { contentType: 'text' })
      return inside
    }

    function inside(code: Code): State | undefined {
      if (code === codes.eof || markdownLineEnding(code)) return nok(code)
      if (code === codes.rightCurlyBrace && codeFence === 0) {
        return effects.check(closingBraces, close, consume)(code)
      }
      if (code === codes.leftCurlyBrace && codeFence === 0)
        return effects.check(openingBraces, nok, consume)(code)
      if (code === codes.backslash && codeFence === 0) {
        if (prefix.length < 'sidenotes'.length) prefix += '\\'
        effects.consume(code)
        return escaped
      }
      if (code === codes.graveAccent) {
        tickRun = 0
        return ticks(code)
      }
      return consume(code)
    }

    function consume(code: Code): State | undefined {
      if (code === codes.eof) return nok(code)
      if (code > 0 && !/\s/.test(String.fromCharCode(code))) hasContent = true
      if (prefix.length < 'sidenotes'.length && code > 0 && (prefix.length > 0 || hasContent)) {
        prefix += String.fromCharCode(code)
        if (prefix === 'sidenotes') return nok(code)
      }
      effects.consume(code)
      return inside
    }

    function escaped(code: Code): State | undefined {
      if (code === codes.eof || markdownLineEnding(code)) return nok(code)
      effects.consume(code)
      hasContent = true
      return inside
    }

    function ticks(code: Code): State | undefined {
      if (code === codes.graveAccent) {
        if (prefix.length < 'sidenotes'.length) prefix += '`'
        tickRun++
        hasContent = true
        effects.consume(code)
        return ticks
      }
      if (codeFence === 0) codeFence = tickRun
      else if (codeFence === tickRun) codeFence = 0
      return inside(code)
    }

    function close(code: Code): State | undefined {
      if (!hasContent) return nok(code)
      effects.exit('chunkText')
      effects.enter('speechMarker')
      effects.consume(code)
      return closed
    }

    function closed(code: Code): State | undefined {
      if (code !== codes.rightCurlyBrace) return nok(code)
      effects.consume(code)
      effects.exit('speechMarker')
      effects.exit('speechPhrase')
      return ok
    }
  }

  return { text: { [codes.leftCurlyBrace]: { name: 'speechPhrase', tokenize } } }
}

const openingBraces: Construct = {
  partial: true,
  tokenize(effects, ok, nok) {
    return first

    function first(code: Code): State | undefined {
      if (code !== codes.leftCurlyBrace) return nok(code)
      effects.consume(code)
      return second
    }

    function second(code: Code): State | undefined {
      if (code !== codes.leftCurlyBrace) return nok(code)
      effects.consume(code)
      return ok
    }
  },
}
