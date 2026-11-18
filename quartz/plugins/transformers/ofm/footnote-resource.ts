import type { Code, Construct, State } from 'micromark-util-types'

const space = (code: Code) =>
  code === 32 || code === -1 || code === -2 || code === -3 || code === -4 || code === -5

/** Protect a complete Markdown resource, including balanced destinations and titles. */
export const footnoteResource: Construct = {
  partial: true,
  tokenize(effects, ok, nok) {
    let balance = 0
    let escaped = false
    let titleEnd: Code = null
    return start
    function consume(code: Code, next: State): State {
      effects.consume(code)
      return next
    }
    function finish(code: Code): State | undefined {
      effects.consume(code)
      effects.exit('ofmFootnoteResource')
      return ok
    }
    function start(code: Code): State | undefined {
      if (code !== 40) return nok(code)
      effects.enter('ofmFootnoteResource')
      return consume(code, beforeDestination)
    }
    function beforeDestination(code: Code): State | undefined {
      if (code === null) return nok(code)
      if (space(code)) return consume(code, beforeDestination)
      if (code === 41) return finish(code)
      if (code === 60) return consume(code, enclosedDestination)
      return rawDestination(code)
    }
    function enclosedDestination(code: Code): State | undefined {
      if (code === null || code === -3 || code === -4 || code === -5) return nok(code)
      if (!escaped) {
        if (code === 60) return nok(code)
        if (code === 62) return consume(code, destinationEnd)
      }
      escaped = code === 92 && !escaped
      return consume(code, enclosedDestination)
    }
    function rawDestination(code: Code): State | undefined {
      if (code === null) return nok(code)
      if (!escaped) {
        if (space(code)) return balance === 0 ? consume(code, afterDestination) : nok(code)
        if (code === 60 || code < 0 || code < 32 || code === 127) return nok(code)
        if (code === 40) balance++
        if (code === 41) {
          if (balance === 0) return finish(code)
          balance--
        }
      }
      escaped = code === 92 && !escaped
      return consume(code, rawDestination)
    }
    function destinationEnd(code: Code): State | undefined {
      if (code === 41) return finish(code)
      return space(code) ? consume(code, afterDestination) : nok(code)
    }
    function afterDestination(code: Code): State | undefined {
      if (code === null) return nok(code)
      if (space(code)) return consume(code, afterDestination)
      if (code === 41) return finish(code)
      if (code !== 34 && code !== 39 && code !== 40) return nok(code)
      titleEnd = code === 40 ? 41 : code
      return consume(code, title)
    }
    function title(code: Code): State | undefined {
      if (code === null) return nok(code)
      if (!escaped) {
        if (code === titleEnd) return consume(code, afterTitle)
        if (titleEnd === 41 && code === 40) return nok(code)
      }
      escaped = code === 92 && !escaped
      return consume(code, title)
    }
    function afterTitle(code: Code): State | undefined {
      if (code === 41) return finish(code)
      return space(code) ? consume(code, afterTitle) : nok(code)
    }
  },
}

declare module 'micromark-util-types' {
  interface TokenTypeMap {
    ofmFootnoteResource: 'ofmFootnoteResource'
  }
}
