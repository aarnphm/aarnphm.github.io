import type { Code, Construct, State, Token } from 'micromark-util-types'

const character =
  /^[-_\p{L}\p{N}\p{M}\p{Extended_Pictographic}\p{Regional_Indicator}\p{Emoji_Modifier}\u200d]$/u
const bodyCode = (code: Code) =>
  code !== null &&
  ((code >= 0xd800 && code <= 0xdfff) || (code >= 0 && character.test(String.fromCharCode(code))))

export const tagConstruct: Construct = {
  name: 'ofmTag',
  tokenize(effects, ok, nok) {
    const previous = this.previous
    let token: Token
    const serialize = (value: Token) => this.sliceSerialize(value)
    return start
    function start(code: Code): State | undefined {
      if (previous === 35 || bodyCode(previous)) return nok(code)
      token = effects.enter('ofmTag')
      effects.consume(code)
      return bodyStart
    }
    function bodyStart(code: Code): State | undefined {
      return bodyCode(code) ? inside(code) : nok(code)
    }
    function inside(code: Code): State | undefined {
      if (bodyCode(code)) {
        effects.consume(code)
        return inside
      }
      if (code === 47) return effects.check(segment, slash, finish)(code)
      return finish(code)
    }
    function slash(code: Code): State | undefined {
      effects.consume(code)
      return inside
    }
    function finish(code: Code): State | undefined {
      effects.exit('ofmTag')
      const value = serialize(token).slice(1)
      return /^[\p{N}/]+$/u.test(value) ? nok(code) : ok(code)
    }
  },
}

const segment: Construct = {
  partial: true,
  tokenize(effects, ok, nok) {
    return start
    function start(code: Code): State | undefined {
      effects.enter('ofmTagSlash')
      effects.consume(code)
      return next
    }
    function next(code: Code): State | undefined {
      effects.exit('ofmTagSlash')
      return bodyCode(code) ? ok(code) : nok(code)
    }
  },
}

declare module 'micromark-util-types' {
  interface TokenTypeMap {
    ofmTag: 'ofmTag'
    ofmTagSlash: 'ofmTagSlash'
  }
}
