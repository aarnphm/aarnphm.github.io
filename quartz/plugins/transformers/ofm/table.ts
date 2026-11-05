import type { Root } from 'mdast'
import type { Code, Construct, Effects, State, Tokenizer } from 'micromark-util-types'
import type { Plugin } from 'unified'
import { tokenizeReference } from '../../../util/references/lexer'
import { inlineComment, inlineFootnote } from './dialect-syntax'

export interface TableReferenceOptions {
  wikilinks: boolean
  comments: boolean
  enableInlineFootnotes: boolean
}

/** Keep authored pipes inside complete inline constructs in their GFM table cell. */
export const remarkTableReferences: Plugin<[TableReferenceOptions], Root> = function (options) {
  const parser = this.parser
  if (!parser) return
  const data = this.data()
  let prepared = false
  this.parser = (document, file) => {
    if (!prepared) {
      // Transformer plugin order can attach GFM after OFM; parsing starts after all attachers.
      data.micromarkExtensions = data.micromarkExtensions?.map(extension => {
        const constructs = extension.flow?.null
        if (!constructs) return extension
        const list = Array.isArray(constructs) ? constructs : [constructs]
        return {
          ...extension,
          flow: {
            ...extension.flow,
            null: list.map(construct =>
              construct.name === 'table'
                ? { ...construct, tokenize: protectTablePipes(construct.tokenize, options) }
                : construct,
            ),
          },
        }
      })
      prepared = true
    }
    return parser(document, file)
  }
}

function protectTablePipes(tokenize: Tokenizer, options: TableReferenceOptions): Tokenizer {
  return function (effects, ok, nok) {
    let protectedUntil = -1
    let literalUntil = -1
    let escaped = false
    let current: Code
    let complete = false
    const tableEffects: Effects = {
      ...effects,
      consume(code) {
        // The GFM state sees cell data; micromark still consumes the authored pipe.
        const original = current === 124 && code === 65 ? current : code
        effects.consume(original)
        escaped = original === 92 ? !escaped : false
      },
    }
    let state: State | undefined = tokenize.call(
      this,
      tableEffects,
      code => {
        complete = true
        return ok(code)
      },
      code => {
        complete = true
        return nok(code)
      },
    )
    const step: State = code => {
      current = code
      if (!escaped && this.now().offset >= Math.max(protectedUntil, literalUntil)) {
        const literalName = code === 96 ? 'codeText' : code === 60 ? 'htmlText' : undefined
        const literalConstructs = code === null ? undefined : this.parser.constructs.text[code]
        const literalConstruct = literalName
          ? (Array.isArray(literalConstructs) ? literalConstructs : [literalConstructs]).find(
              construct => construct?.name === literalName,
            )
          : undefined
        const tokenizer =
          literalConstruct?.tokenize ??
          (code === 91 && options.wikilinks
            ? tokenizeReference
            : code === 94 && options.enableInlineFootnotes
              ? inlineFootnote
              : code === 37 && options.comments
                ? inlineComment
                : undefined)
        if (tokenizer) {
          const check: Construct = {
            partial: true,
            tokenize: withinRow(tokenizer, end => {
              if (literalConstruct) literalUntil = end
              else protectedUntil = end
            }),
          }
          return effects.check(check, advance, advance)(code)
        }
      }
      return advance(code)
    }

    const advance: State = code => {
      const input = code === 124 && this.now().offset < protectedUntil ? 65 : code
      state = state?.(input)
      return complete ? state : state ? step : undefined
    }
    return step
  }
}

function withinRow(tokenize: Tokenizer, acceptEnd: (end: number) => void): Tokenizer {
  return function (effects, ok, nok) {
    let complete = false
    const accept: State = code => {
      complete = true
      acceptEnd(this.now().offset)
      return ok(code)
    }
    const reject: State = code => {
      complete = true
      return nok(code)
    }
    let state = tokenize.call(this, effects, accept, reject)
    const step: State = code => {
      if (code === null || code === -5 || code === -4 || code === -3) return reject(code)
      const next = state(code)
      if (complete || next === accept) return next
      if (!next) return undefined
      state = next
      return step
    }
    return step
  }
}
