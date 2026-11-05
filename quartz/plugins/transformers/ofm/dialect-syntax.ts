import type { Code, Construct, Effects, Extension, State, Tokenizer } from 'micromark-util-types'
import { footnoteResource } from './footnote-resource'
import { highlightConstruct } from './highlight-syntax'
import { inlineChunks } from './inline-chunks'
import { paragraphWithComment } from './paragraph-comments'
import { tagConstruct } from './tag-syntax'

const lineEnding = (code: Code) => code === -5 || code === -4 || code === -3

function commentData(effects: Effects) {
  let open = false
  const close = () => {
    if (open) effects.exit('ofmCommentData')
    open = false
  }
  return {
    close,
    consume(code: Code) {
      if (lineEnding(code)) {
        close()
        effects.enter('lineEnding')
        effects.consume(code)
        effects.exit('lineEnding')
      } else {
        if (!open) effects.enter('ofmCommentData')
        open = true
        effects.consume(code)
      }
    },
  }
}

export const inlineComment: Tokenizer = function (effects, ok, nok) {
  let escaped = false
  const data = commentData(effects)
  return start
  function start(code: Code): State | undefined {
    effects.enter('ofmComment')
    data.consume(code)
    return second
  }
  function second(code: Code): State | undefined {
    if (code !== 37) return nok(code)
    data.consume(code)
    return inside
  }
  function inside(code: Code): State | undefined {
    if (code === null) return nok(code)
    if (code === 37 && !escaped) {
      data.consume(code)
      return close
    }
    escaped = code === 92 && !escaped
    data.consume(code)
    return inside
  }
  function close(code: Code): State | undefined {
    if (code === 37) {
      data.consume(code)
      data.close()
      effects.exit('ofmComment')
      return ok
    }
    return inside(code)
  }
}

const blockComment: Tokenizer = function (effects, ok, nok) {
  const interrupt = this.interrupt
  let previousPercent = false
  const data = commentData(effects)
  return start
  function start(code: Code): State | undefined {
    effects.enter('ofmComment')
    data.consume(code)
    return second
  }
  function second(code: Code): State | undefined {
    if (code !== 37) return nok(code)
    data.consume(code)
    return openingLine
  }
  function openingLine(code: Code): State | undefined {
    if (code === 32 || code === -1 || code === -2) {
      data.consume(code)
      return openingLine
    }
    if (code === null) return finish(code)
    if (!lineEnding(code)) return nok(code)
    if (interrupt) return ok(code)
    data.consume(code)
    return inside
  }
  function inside(code: Code): State | undefined {
    if (code === null) return finish(code)
    data.consume(code)
    if (code === 37 && previousPercent) return closingLine
    previousPercent = code === 37
    return inside
  }
  function closingLine(code: Code): State | undefined {
    data.close()
    effects.exit('ofmComment')
    return tailStart(code)
  }
  function tailStart(code: Code): State | undefined {
    if (code === null || lineEnding(code)) return ok(code)
    if (code === 32 || code === -1 || code === -2) {
      effects.enter('whitespace')
      effects.consume(code)
      effects.exit('whitespace')
      return tailStart
    }
    effects.enter('ofmCommentTail')
    effects.enter('chunkText', { contentType: 'text' })
    return tailInside(code)
  }
  function tailInside(code: Code): State | undefined {
    if (code === null || lineEnding(code)) {
      effects.exit('chunkText')
      effects.exit('ofmCommentTail')
      return ok(code)
    }
    effects.consume(code)
    return tailInside
  }
  function finish(code: Code): State | undefined {
    data.close()
    effects.exit('ofmComment')
    return ok(code)
  }
}

export const inlineFootnote: Tokenizer = function (effects, ok, nok) {
  const offset = () => this.now().offset
  const syntax = this.parser.constructs
  let depth = 0
  let escaped = false
  let protectedUntil = -1
  const chunks = inlineChunks(effects)
  let afterLabel = false
  return start
  function start(code: Code): State | undefined {
    effects.enter('ofmInlineFootnote')
    effects.enter('ofmFootnoteMarker')
    effects.consume(code)
    return open
  }
  function open(code: Code): State | undefined {
    if (code !== 91) return nok(code)
    effects.consume(code)
    effects.exit('ofmFootnoteMarker')
    effects.enter('ofmFootnoteBody')
    return bodyStart
  }
  function bodyStart(code: Code): State | undefined {
    if (code === null) return nok(code)
    if (code === 93) return close(code)
    return inside(code)
  }
  function inside(code: Code): State | undefined {
    if (code === null) return nok(code)
    if (!escaped && offset() >= protectedUntil) {
      if (afterLabel && code === 40) {
        return effects.check(
          {
            ...footnoteResource,
            tokenize(checkEffects, checkOk, checkNok) {
              return footnoteResource.tokenize.call(
                this,
                checkEffects,
                next => {
                  protectedUntil = this.now().offset
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
      const name =
        code === 96
          ? 'codeText'
          : code === 36
            ? 'mathText'
            : code === 60
              ? 'htmlText'
              : code === 37
                ? 'ofmComment'
                : undefined
      const available = syntax.text[code]
      const constructs = Array.isArray(available) ? available : available ? [available] : []
      const construct = name ? constructs.find(item => item.name === name) : undefined
      if (construct) {
        return effects.check(
          {
            partial: true,
            tokenize(checkEffects, checkOk, checkNok) {
              return construct.tokenize.call(
                this,
                checkEffects,
                next => {
                  protectedUntil = this.now().offset
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
    let closesLabel = false
    if (!escaped && offset() >= protectedUntil) {
      if (code === 91) depth++
      if (code === 93) {
        if (depth === 0) return close(code)
        depth--
        closesLabel = true
      }
    }
    afterLabel = closesLabel
    escaped = code === 92 && !escaped
    chunks.consume(code)
    return inside
  }
  function close(code: Code): State | undefined {
    chunks.close()
    effects.exit('ofmFootnoteBody')
    effects.enter('ofmFootnoteMarker')
    effects.consume(code)
    effects.exit('ofmFootnoteMarker')
    effects.exit('ofmInlineFootnote')
    return ok
  }
}

export function dialectSyntax(
  options: {
    comments: boolean
    enableInlineFootnotes: boolean
    highlight?: boolean
    parseTags?: boolean
  },
  hasComments: () => boolean = () => true,
): Extension {
  const paragraph: Construct = {
    ...paragraphWithComment,
    tokenize(effects, ok, nok) {
      return hasComments() ? paragraphWithComment.tokenize.call(this, effects, ok, nok) : nok
    },
  }
  return {
    flow: options.comments
      ? { 37: { name: 'ofmBlockComment', tokenize: blockComment, concrete: true }, null: paragraph }
      : {},
    ...(options.highlight
      ? { insideSpan: { null: [highlightConstruct] }, attentionMarkers: { null: [61] } }
      : {}),
    text: {
      ...(options.parseTags ? { 35: tagConstruct } : {}),
      ...(options.highlight ? { 61: highlightConstruct } : {}),
      ...(options.comments ? { 37: { name: 'ofmComment', tokenize: inlineComment } } : {}),
      ...(options.enableInlineFootnotes
        ? { 94: { name: 'ofmInlineFootnote', tokenize: inlineFootnote } }
        : {}),
    },
  }
}

declare module 'micromark-util-types' {
  interface TokenTypeMap {
    ofmHighlightTemporary: 'ofmHighlightTemporary'
    ofmHighlightText: 'ofmHighlightText'
    ofmHighlight: 'ofmHighlight'
    ofmHighlightMarker: 'ofmHighlightMarker'
    ofmComment: 'ofmComment'
    ofmCommentData: 'ofmCommentData'
    ofmCommentTail: 'ofmCommentTail'
    ofmInlineFootnote: 'ofmInlineFootnote'
    ofmFootnoteMarker: 'ofmFootnoteMarker'
    ofmFootnoteBody: 'ofmFootnoteBody'
  }
}
