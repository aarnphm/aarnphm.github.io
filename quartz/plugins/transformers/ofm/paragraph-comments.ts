import type { Code, Construct, State, Token, Tokenizer } from 'micromark-util-types'

const name = 'ofmCommentParagraph'
const lineEnding = (code: Code) => code === -5 || code === -4 || code === -3
const space = (code: Code) => code === 32 || code === -1 || code === -2
const blankLine = /(?:\r\n|\r|\n)[\t ]*(?:\r\n|\r|\n)/

const tokenize: Tokenizer = function (effects, ok, nok) {
  const parser = this.parser
  const interrupt = this.interrupt
  const position = () => this.now()
  const serialize = (token: Pick<Token, 'start' | 'end'>, expandTabs?: boolean) =>
    this.sliceSerialize(token, expandTabs)
  const nullable = parser.constructs.flow.null
  const flow = {
    ...parser.constructs.flow,
    null: (Array.isArray(nullable) ? nullable : nullable ? [nullable] : []).filter(
      construct => construct.name !== name,
    ),
  }
  let previous: Token | undefined
  let literalUntil = -1
  let commentUntil = -1
  let closedAcrossBlank = false
  let escaped = false

  const continuation: Construct = {
    partial: true,
    tokenize(checkEffects, checkOk, checkNok) {
      let prefix: Token | undefined
      return start
      function start(code: Code): State | undefined {
        checkEffects.exit('chunkContent')
        checkEffects.enter('lineEnding')
        checkEffects.consume(code)
        checkEffects.exit('lineEnding')
        return prefixStart
      }
      function prefixStart(code: Code): State | undefined {
        if (!space(code)) return prefixed(code)
        prefix = checkEffects.enter('linePrefix')
        return prefixInside(code)
      }
      function prefixInside(code: Code): State | undefined {
        if (space(code)) {
          checkEffects.consume(code)
          return prefixInside
        }
        checkEffects.exit('linePrefix')
        return prefixed(code)
      }
      function prefixed(code: Code): State | undefined {
        if (code === null || lineEnding(code)) return checkNok(code)
        if (
          prefix &&
          !parser.constructs.disable.null?.includes('codeIndented') &&
          serialize(prefix, true).length >= 4
        )
          return checkOk(code)
        return checkEffects.interrupt(flow, checkNok, checkOk)(code)
      }
    },
  }

  return start
  function start(code: Code): State | undefined {
    if (interrupt) return nok(code)
    effects.enter('content')
    previous = effects.enter('chunkContent', { contentType: 'content' })
    return inside(code)
  }
  function inside(code: Code): State | undefined {
    if (code === null) return finish(code)
    if (lineEnding(code)) {
      return position().offset < commentUntil
        ? nextLine(code)
        : effects.check(continuation, nextLine, finish)(code)
    }
    if (
      !escaped &&
      (code === 37 || code === 96 || code === 36 || code === 60 || code === 91 || code === 33) &&
      position().offset >= Math.max(literalUntil, commentUntil)
    ) {
      const tokenName =
        code === 37
          ? 'ofmComment'
          : code === 96
            ? 'codeText'
            : code === 36
              ? 'mathText'
              : code === 60
                ? 'htmlText'
                : code === 91 || code === 33
                  ? 'wikilink'
                  : undefined
      const available = parser.constructs.text[code]
      const constructs = Array.isArray(available) ? available : available ? [available] : []
      const construct = tokenName ? constructs.find(item => item.name === tokenName) : undefined
      if (construct) {
        const start = position()
        return effects.check(
          {
            partial: true,
            tokenize(checkEffects, checkOk, checkNok) {
              return construct.tokenize.call(
                this,
                checkEffects,
                next => {
                  const end = this.now()
                  const spansBlank = blankLine.test(this.sliceSerialize({ start, end }))
                  if (tokenName === 'ofmComment') {
                    commentUntil = end.offset
                    closedAcrossBlank ||= spansBlank
                  } else {
                    if (spansBlank) return checkNok(next)
                    literalUntil = end.offset
                  }
                  return checkOk(next)
                },
                checkNok,
              )
            },
          },
          consume,
          consume,
        )(code)
      }
    }
    return consume(code)
  }
  function consume(code: Code): State | undefined {
    escaped = code === 92 && !escaped
    effects.consume(code)
    return inside
  }
  function nextLine(code: Code): State | undefined {
    effects.consume(code)
    effects.exit('chunkContent')
    const current = effects.enter('chunkContent', { contentType: 'content', previous })
    if (previous) previous.next = current
    previous = current
    escaped = false
    return inside
  }
  function finish(code: Code): State | undefined {
    effects.exit('chunkContent')
    effects.exit('content')
    return closedAcrossBlank ? ok(code) : nok(code)
  }
}

/** Keep a paired comment's blank lines inside the same upstream content stream. */
export const paragraphWithComment: Construct = { name, concrete: true, tokenize }
